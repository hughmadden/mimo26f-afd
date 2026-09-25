//! The A8 `Engine` implementation — the coordinator wraps the serving forward
//! behind the OpenAI-compatible API's `Engine` trait (tokenize / render_chat /
//! generate). Only compiled under `cuda` (generate needs the GPU serving path).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use mimo26_api::engine::{Engine, GenerateOutcome, GenerateParams, ImageInput, IMAGE_CLOSE, IMAGE_OPEN};
use mimo26_api::types::{ChatMessage, Tool, ToolCall};

use crate::config::Config;
use crate::dforward::{device_free_bytes, DeviceForward, DeviceKv, EmbedOverlay, BATCH_ROWS, DFLASH_DRAFTS, KV_MARGIN_BYTES};
use crate::hostcache::{HostCache, Kind};
use mimo26_attn::device::DeviceBuffer;
use crate::serving::{Fp8KvCache, ServingModel};
use crate::streaming::flush_pending;
use crate::tokenizer::BpeTokenizer;
use crate::vision::VisionTower;
use crate::wire::WireClient;
use crate::{greedy, Role};

/// Map the API's flat role string to the coordinator's role enum.
fn map_role(role: &str) -> Role {
    match role {
        "assistant" => Role::Assistant,
        "system" => Role::System,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

/// Call the API delta callback, catching a panic (the SSE write path can panic
/// on bad input, e.g. a multi-byte char at a slice boundary) so the request ends
/// cleanly instead of unwinding through the held engine locks and poisoning them
/// (D5).
fn emit_delta(on_delta: &mut dyn FnMut(&str), text: &str) -> Result<(), String> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_delta(text)));
    r.map_err(|_| "on_delta panicked".to_string())
}

/// Map one wire tool call to the renderer's ordered-argument shape. The
/// arguments string is parsed to JSON and its pairs are kept in the CLIENT's
/// insertion order (D2b item 3: vLLM `json.loads` the arguments string before
/// templating, so the template iterates the client's JSON order — never the
/// tool schema's declared order).
fn map_tool_call(tc: &ToolCall) -> crate::chat::ChatToolCall {
    let parsed = mimo26_api::json::parse(&tc.arguments).unwrap_or(mimo26_api::json::Json::Null);
    let arguments: Vec<(String, mimo26_api::json::Json)> = match parsed {
        mimo26_api::json::Json::Object(o) => o,
        _ => Vec::new(),
    };
    crate::chat::ChatToolCall { name: tc.name.clone(), arguments }
}

/// Map API messages to the coordinator's render shape, carrying assistant tool
/// calls so the renderer emits them per the checkpoint template.
fn map_messages(messages: &[ChatMessage]) -> Vec<crate::chat::ChatMessage> {
    messages
        .iter()
        .map(|m| crate::chat::ChatMessage {
            role: map_role(&m.role),
            content: m.content.clone(),
            reasoning: None,
            tool_calls: if m.tool_calls.is_empty() {
                None
            } else {
                Some(m.tool_calls.iter().map(map_tool_call).collect())
            },
        })
        .collect()
}

/// The serving forward behind the engine: the device-resident forward (perf
/// reset R1, the default) behind the batching scheduler (W4), or the
/// host-orchestrated reference path (one request at a time).
enum Backend {
    Host { model: ServingModel, caches: Mutex<Vec<Fp8KvCache>>, wire: Mutex<WireClient> },
    Device { jobs: Mutex<mpsc::Sender<Job>> },
}

/// The coordinator engine: a tokenizer plus the serving backend. The `Engine`
/// trait is `&self`; the device backend batches concurrent requests.
pub struct CoordinatorEngine {
    cfg: Config,
    backend: Backend,
    tok: BpeTokenizer,
    /// The longest request one slot can hold with the rest of the pool idle
    /// (measured at startup from free GPU memory; device backend only).
    max_context: Option<usize>,
    /// `<|vision_start|>` and `<|vision_end|>` when the image encoder is loaded (perf reset V2).
    vision_tokens: Option<(u32, u32)>,
}

/// One generation for the scheduler: the prompt, the token budget, the channel
/// each sampled token (or an error) goes back on, and the caller's cancel flag.
struct Job {
    ids: Vec<usize>,
    max_tokens: usize,
    tx: mpsc::Sender<Result<usize, String>>,
    cancel: Arc<AtomicBool>,
    images: Vec<ImageSpan>,
}

/// An image in a prompt (perf reset V2): the prompt's image tokens `[start, start + tokens)`
/// take the encoder's rows for `image`.
struct ImageSpan {
    start: usize,
    image: Arc<ImageInput>,
}

/// The token id standing for row `i` of the image with hash `hash` (perf reset V2): past any
/// vocabulary (bit 31 set), and a function of the image's bytes, so the prefix caches, which
/// compare token ids, share a prompt only when its images are the same.
pub fn image_token_id(hash: u64, i: usize) -> u32 {
    let mut x = hash ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    0x8000_0000 | (x as u32 & 0x7fff_ffff)
}

/// Encode a rendered prompt whose images stand as markers (`IMAGE_OPEN` hash `:` tokens
/// `IMAGE_CLOSE`, placed by the API): the text between them is encoded as usual, and each marker
/// becomes `<|vision_start|>`, its image token ids, `<|vision_end|>` — what the chat template's
/// `<|vision_start|><|image_pad|><|vision_end|>` encodes to once the processor expands the pad.
/// Special tokens split the text, so encoding the pieces apart matches encoding the whole.
fn encode_marked(tok: &BpeTokenizer, vision: Option<(u32, u32)>, text: &str) -> Vec<u32> {
    let Some((start, end)) = vision.filter(|_| text.contains(IMAGE_OPEN)) else { return tok.encode(text) };
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(IMAGE_OPEN) {
        let after = &rest[i + IMAGE_OPEN.len_utf8()..];
        let Some(j) = after.find(IMAGE_CLOSE) else { break };
        let parsed = after[..j].split_once(':').and_then(|(h, n)| Some((u64::from_str_radix(h, 16).ok()?, n.parse::<usize>().ok()?)));
        let Some((hash, n)) = parsed else { break };
        if i > 0 {
            ids.extend(tok.encode(&rest[..i]));
        }
        ids.push(start);
        ids.extend((0..n).map(|k| image_token_id(hash, k)));
        ids.push(end);
        rest = &after[j + IMAGE_CLOSE.len_utf8()..];
    }
    if !rest.is_empty() {
        ids.extend(tok.encode(rest));
    }
    ids
}

/// Where each image of a request sits in its prompt: the runs of image token ids, matched in
/// order against the request's images by length and first id.
fn image_spans(ids: &[usize], images: &[Arc<ImageInput>]) -> Result<Vec<ImageSpan>, String> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] < 0x8000_0000 {
            i += 1;
            continue;
        }
        let start = i;
        while i < ids.len() && ids[i] >= 0x8000_0000 {
            i += 1;
        }
        let Some(img) = images.get(spans.len()) else { return Err("the prompt has more images than the request".into()) };
        if i - start != img.tokens || ids[start] != image_token_id(img.hash, 0) as usize {
            return Err(format!("image {} does not match its place in the prompt", spans.len()));
        }
        spans.push(ImageSpan { start, image: img.clone() });
    }
    if spans.len() != images.len() {
        return Err(format!("{} images in the request, {} in the prompt", images.len(), spans.len()));
    }
    Ok(spans)
}

/// Sets a job's cancel flag when the caller stops reading (stop string, client
/// gone, error), so the scheduler frees its slot at the next step.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// A request being decoded: its cache slot and progress.
struct Active {
    kv: DeviceKv,
    last: usize,
    generated: usize,
    max: usize,
    tx: mpsc::Sender<Result<usize, String>>,
    cancel: Arc<AtomicBool>,
    /// The prompt then every token sent (the KV holds all but the last).
    hist: Vec<usize>,
    /// Retained snapshots inside this slot's history.
    points: Vec<Point>,
}

/// A request whose prompt is still being prefilled (perf reset Q1): one segment
/// per scheduler round, a decode step for the running requests between
/// segments, so a long prompt does not stall every other stream.
struct Prefilling {
    kv: DeviceKv,
    ids: Vec<usize>,
    /// Prompt tokens in the KV so far.
    done: usize,
    /// The greedy token after `done` (from the last segment's logits).
    next: Option<usize>,
    points: Vec<Point>,
    max: usize,
    tx: mpsc::Sender<Result<usize, String>>,
    cancel: Arc<AtomicBool>,
    /// The prompt's images (perf reset V2) and, once a segment needs them, their encoded rows.
    images: Vec<ImageSpan>,
    overlay: Option<EmbedOverlay>,
}

/// A retained snapshot on the device (perf reset K3, the design's device banks):
/// an exact position in a slot's token history. The slot's GA rows `[0, len)`
/// are the snapshot's (append-only); the SWA/draft state at `len` is saved in
/// `state` (`DeviceKv::save_state_dev`, 35 MB for MiMo). Saving one is a
/// device-to-device copy, never RAM traffic.
struct Point {
    len: usize,
    /// The greedy token after the snapshot (an exact hit needs no forward).
    next: usize,
    kind: Kind,
    state: DeviceBuffer,
    swa_rows: usize,
    last_use: u64,
}

impl Point {
    /// `kv`'s current position as a point; none when the device has no room.
    fn save(kv: &DeviceKv, next: usize, kind: Kind, now: u64) -> Option<Point> {
        let state = DeviceBuffer::alloc(kv.state_bytes()).ok()?;
        match kv.save_state_dev(&state) {
            Ok(swa_rows) => Some(Point { len: kv.tokens(), next, kind, state, swa_rows, last_use: now }),
            Err(e) => {
                eprintln!("[coordinator] snapshot save failed: {e}");
                None
            }
        }
    }

    /// Put `kv`, which holds this point's GA rows, at this point.
    fn load(&self, kv: &mut DeviceKv) -> Result<(), String> {
        kv.rewind_ga(self.len)?;
        kv.load_state_dev(&self.state, self.swa_rows)
    }
}

/// A finished request's slot kept on the device for its points (no copies).
struct Retained {
    kv: DeviceKv,
    /// The tokens of the slot's GA rows (at least up to its last point).
    hist: Vec<usize>,
    points: Vec<Point>,
}

impl Retained {
    fn last_use(&self) -> u64 {
        self.points.iter().map(|p| p.last_use).max().unwrap_or(0)
    }
}

/// Snapshots shorter than this are neither retained nor stored (recomputing
/// them is cheap; the design's `--host-cache-min-tokens`).
const MIN_RETAIN: usize = 512;

/// The encoded rows of the images of `p` that its remaining prefill reaches (perf reset V2).
fn encode_images(pool: &mut Pool, vision: Option<&VisionTower>, p: &Prefilling) -> Result<EmbedOverlay, String> {
    let tower = vision.ok_or("this server has no image encoder loaded")?;
    let spans: Vec<&ImageSpan> = p.images.iter().filter(|s| s.start + s.image.tokens > p.done).collect();
    let most = spans.iter().map(|s| s.image.tokens * 4).max().unwrap_or(0);
    pool.make_room_bytes(tower.weight_bytes() + VisionTower::scratch_bytes(most))?;
    let t0 = std::time::Instant::now();
    let dev = tower.upload()?;
    let mut overlay = EmbedOverlay::default();
    for span in spans {
        let img = &span.image;
        let t1 = std::time::Instant::now();
        let rgb = mimo26_image::RgbImage { width: img.width, height: img.height, data: img.rgb.clone() };
        let patches = mimo26_image::preprocess(&rgb).map_err(|e| format!("image: {e}"))?;
        let n = (patches.grid_h * patches.grid_w) as usize;
        if n / 4 != img.tokens {
            return Err(format!("image grid {}x{} gives {} tokens, the prompt has {}", patches.grid_h, patches.grid_w, n / 4,
                img.tokens));
        }
        let t2 = std::time::Instant::now();
        let (rows, _) = dev.encode(&patches.data, patches.grid_h as usize, patches.grid_w as usize, false)?;
        eprintln!("[vision] {}x{} image, {} tokens: preprocess {:.0} ms, encode {:.0} ms", img.width, img.height,
            img.tokens, (t2 - t1).as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3);
        overlay.spans.push((span.start, rows));
    }
    eprintln!("[vision] {} image(s) in {:.0} ms (weights upload included)", overlay.spans.len(), t0.elapsed().as_secs_f64() * 1e3);
    Ok(overlay)
}

/// GA rows reserved at admission beyond the prompt: the first output tokens plus
/// a verify block (ARCHITECTURE §11.1: prompt + min(max_tokens, 8192), growing
/// per round after that). At least 1,024 output rows, so a short follow-up turn
/// resumes in its retained slot without growing it (a growth near the top of
/// memory needs a whole layer's buffers twice over).
fn admit_rows(prompt: usize, max_tokens: usize) -> usize {
    prompt + max_tokens.clamp(1024, 8192) + 64
}

/// Can `kv` take a request of `rows` GA rows now? Needs the GA growth, its
/// transient (one layer's buffers held twice while it copies), and the
/// prefill's SWA working set, over [`KV_MARGIN_BYTES`] of slack.
fn admit_check(kv: &DeviceKv, rows: usize) -> Result<(), String> {
    let cap = kv.ga_capacity();
    let grow = if rows > cap { rows.next_multiple_of(4096) - cap } else { 0 } * kv.ga_token_bytes();
    let need = grow + kv.ga_grow_transient(rows) + kv.swa_prefill_bytes() + KV_MARGIN_BYTES;
    let free = device_free_bytes()?;
    if need > free {
        return Err(format!(
            "KV pool: a {rows}-token request needs {:.2} GiB of GPU memory, {:.2} GiB is free; retry when other \
             requests finish",
            need as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64
        ));
    }
    Ok(())
}

/// The KV slots and snapshot tiers (perf reset K3, the DS41RT host cache design
/// v3 in its `on-evict` store mode): free slots; retained slots holding points
/// (the device tier: no copies while nothing is under pressure); the RAM tier
/// behind them. A point goes to RAM only when the device evicts it: its bank
/// over `bank` points, or its slot or the slot's memory needed by a request.
struct Pool {
    free: Vec<DeviceKv>,
    retained: Vec<Retained>,
    cache: Option<HostCache>,
    clock: u64,
    /// Points kept on the device per bank (the design's `--prefix-cache-entries`).
    bank: usize,
}

impl Pool {
    fn now(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Store point `p` of the slot `kv` (holding `hist`) to RAM, when the RAM tier is on.
    fn store(cache: &mut Option<HostCache>, kv: &DeviceKv, hist: &[usize], p: &Point) {
        if let Some(c) = cache.as_mut() {
            if let Err(e) = c.capture(kv, &hist[..p.len], p.next, p.kind, &p.state, p.swa_rows) {
                eprintln!("[hostcache] store failed: {e}");
            }
        }
    }

    /// Give a slot back: every row dropped, GA and SWA shrunk to the build size.
    fn release(&mut self, mut kv: DeviceKv) {
        if let Err(e) = kv.shrink_ga().and_then(|_| kv.shrink_swa()) {
            eprintln!("[coordinator] KV shrink failed: {e}");
        }
        self.free.push(kv);
    }

    /// Evict the least recently used retained slot: its points to RAM, its slot
    /// back to the free list. False when none is left.
    fn evict_lru(&mut self) -> bool {
        let Some(i) = (0..self.retained.len()).min_by_key(|&i| self.retained[i].last_use()) else { return false };
        let r = self.retained.swap_remove(i);
        eprintln!("[coordinator] device pressure: evicting a retained {}-token slot ({} snapshots) to RAM",
            r.kv.tokens(), r.points.len());
        for p in &r.points {
            Self::store(&mut self.cache, &r.kv, &r.hist, p);
        }
        self.release(r.kv);
        true
    }

    /// A clean slot: a free one, else the least recently used retained one.
    fn take_free(&mut self) -> Option<DeviceKv> {
        if self.free.is_empty() {
            self.evict_lru();
        }
        self.free.pop().map(|mut kv| {
            kv.reset();
            kv
        })
    }

    /// Room on the device for `kv` to hold `rows` GA rows, evicting retained
    /// slots (least recently used first) while it does not fit.
    fn make_room(&mut self, kv: &DeviceKv, rows: usize) -> Result<(), String> {
        loop {
            match admit_check(kv, rows) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if !self.evict_lru() {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// `bytes` of free device memory beyond the margin, evicting retained slots (least recently
    /// used first) while there is not (perf reset V2: the image encoder's transient buffers).
    fn make_room_bytes(&mut self, bytes: usize) -> Result<(), String> {
        loop {
            let free = device_free_bytes()?;
            if free >= bytes + KV_MARGIN_BYTES {
                return Ok(());
            }
            if !self.evict_lru() {
                return Err(format!("the image encoder needs {} MiB of GPU memory; {} MiB is free", bytes >> 20, free >> 20));
            }
        }
    }

    /// The longest retained point that prefixes `ids`: `(slot, point)`. Equal
    /// lengths prefer a point with nothing after it in its slot (used in place).
    fn device_hit(&self, ids: &[usize]) -> Option<(usize, usize)> {
        let mut best: Option<((usize, bool), usize, usize)> = None;
        for (si, r) in self.retained.iter().enumerate() {
            let common = r.hist.iter().zip(ids).take_while(|(a, b)| a == b).count();
            let top = r.points.iter().map(|p| p.len).max().unwrap_or(0);
            for (pi, p) in r.points.iter().enumerate() {
                let key = (p.len, p.len == top);
                if p.len <= common && best.map_or(true, |b| key > b.0) {
                    best = Some((key, si, pi));
                }
            }
        }
        best.map(|b| (b.1, b.2))
    }

    /// A slot for a new request of `rows` GA rows, resumed at the longest exact
    /// snapshot of `ids` on the device or in RAM (equal lengths: the device).
    /// Returns the slot, the points it carries and where it resumes, `(tokens,
    /// next token)` (none: cold).
    ///
    /// A device hit on a slot's last point continues in that slot. A hit on an
    /// earlier point forks it into a free slot when one is free and fits without
    /// evicting anything (the retained slot stays whole, as the design's
    /// copy-on-write sharing keeps it); otherwise the slot rewinds to the point
    /// and its later points go to RAM.
    fn admit(&mut self, fwd: &DeviceForward, ids: &[usize], rows: usize)
        -> Result<(DeviceKv, Vec<Point>, Option<(usize, usize)>), String> {
        let now = self.now();
        let dev = self.device_hit(ids);
        let dev_len = dev.map_or(0, |(si, pi)| self.retained[si].points[pi].len);
        let host_longer = dev_len < ids.len()
            && self.cache.as_ref().and_then(|c| c.lookup(ids)).is_some_and(|(_, n)| n > dev_len);
        if let (Some((si, pi)), false) = (dev, host_longer) {
            let top = self.retained[si].points.iter().map(|p| p.len).max().unwrap_or(0);
            // Fork: the point's GA rows copied into a free slot that fits as is.
            if dev_len < top {
                if let Some(mut kv) = self.free.pop() {
                    kv.reset();
                    let fits = fwd.attach_draft(&mut kv).is_ok() && admit_check(&kv, rows).is_ok();
                    let r = &mut self.retained[si];
                    let forked = fits
                        && kv
                            .reserve_ga(rows)
                            .and_then(|_| kv.fork_ga(&r.kv, dev_len))
                            .and_then(|_| r.points[pi].load(&mut kv))
                            .map_err(|e| eprintln!("[coordinator] snapshot fork failed: {e}"))
                            .is_ok();
                    if forked {
                        r.points[pi].last_use = now;
                        let next = r.points[pi].next;
                        eprintln!("[coordinator] device hit: {dev_len} of {} prompt tokens ({:?} snapshot, forked)",
                            ids.len(), r.points[pi].kind);
                        return Ok((kv, Vec::new(), Some((dev_len, next))));
                    }
                    self.release(kv);
                }
            }
            // In place: the slot rewinds to the point; later points go to RAM.
            let mut r = self.retained.swap_remove(si);
            let x = &mut r.points[pi];
            x.last_use = now;
            let (next, kind) = (x.next, x.kind);
            let (keep, later): (Vec<Point>, Vec<Point>) = r.points.into_iter().partition(|p| p.len <= dev_len);
            for p in &later {
                Self::store(&mut self.cache, &r.kv, &r.hist, p);
            }
            drop(later);
            let mut kv = r.kv;
            let placed = fwd
                .attach_draft(&mut kv)
                .and_then(|_| self.make_room(&kv, rows))
                .and_then(|_| kv.reserve_ga(rows))
                .and_then(|_| keep.iter().find(|p| p.len == dev_len).expect("the hit point").load(&mut kv));
            match placed {
                Ok(()) => {
                    eprintln!("[coordinator] device hit: {dev_len} of {} prompt tokens ({kind:?} snapshot, in place)",
                        ids.len());
                    return Ok((kv, keep, Some((dev_len, next))));
                }
                // Too little memory to grow the slot in place (near the top the old
                // GA buffers cannot coexist with the grown ones): with the RAM tier
                // on, move it through RAM into a fresh slot of exactly the request's
                // size (below: store, free, restore).
                Err(e) if self.cache.is_some() => {
                    eprintln!("[coordinator] cannot grow a retained {}-token slot in place ({e}); relocating it \
                        through RAM", kv.tokens());
                    for p in &keep {
                        Self::store(&mut self.cache, &kv, &r.hist, p);
                    }
                    drop(keep);
                    self.release(kv);
                }
                Err(e) => {
                    // The slot keeps its points; the request is refused.
                    r.hist.truncate(dev_len);
                    self.retained.push(Retained { kv, hist: r.hist, points: keep });
                    return Err(e);
                }
            }
        }
        // A clean slot (evicting the least recently used retained slot if none is
        // free), then the RAM tier: looked up again, since making room stores to it.
        let mut kv = self.take_free().ok_or("no KV slot")?;
        if let Err(e) = fwd.attach_draft(&mut kv).and_then(|_| self.make_room(&kv, rows)).and_then(|_| kv.reserve_ga(rows)) {
            self.release(kv);
            return Err(e);
        }
        if let Some(c) = self.cache.as_mut() {
            if let Some((i, _)) = c.lookup(ids) {
                match c.restore(i, &mut kv) {
                    Ok((n, next, kind)) => {
                        // The rebuilt snapshot joins the device bank (the design's restore).
                        let points = Point::save(&kv, next, kind, now).into_iter().collect();
                        return Ok((kv, points, Some((n, next))));
                    }
                    Err(e) => {
                        eprintln!("[hostcache] restore failed, prefilling cold: {e}");
                        kv.reset();
                    }
                }
            }
        }
        Ok((kv, Vec::new(), None))
    }

    /// Retire a finished request: its slot stays on the device with its points
    /// plus a completion-end turn point (device copies only), or is freed when it
    /// has none.
    fn retire(&mut self, mut a: Active) {
        let n = a.hist.len() - 1;
        let cancelled = a.cancel.load(Ordering::Relaxed);
        // Back to the history's end: a speculative step may hold more rows.
        let at_n = a.kv.tokens() == n || (a.kv.tokens() > n && a.kv.truncate(n).is_ok());
        if at_n && !cancelled && n >= MIN_RETAIN && a.points.iter().all(|p| p.len < n) {
            let now = self.now();
            // An identical snapshot already retained is refreshed, not kept twice
            // (the design's radix bank holds one entry per key).
            if let Some((si, pi)) = self.device_hit(&a.hist[..n]).filter(|&(si, pi)| self.retained[si].points[pi].len == n) {
                self.retained[si].points[pi].last_use = now;
            } else if let Some(p) = Point::save(&a.kv, a.hist[n], Kind::Turn, now) {
                a.points.push(p);
            }
        }
        if a.points.is_empty() {
            self.release(a.kv);
            return;
        }
        a.hist.truncate(n);
        self.retained.push(Retained { kv: a.kv, hist: a.hist, points: a.points });
    }

    /// A prefill abandoned at `done` tokens (client gone, or an error after the
    /// forward): its position is kept as a snapshot, so a retry of the same
    /// prompt resumes there instead of prefilling again.
    fn park(&mut self, p: Prefilling) {
        let Prefilling { mut kv, mut ids, done, next, mut points, .. } = p;
        if let Err(e) = kv.shrink_swa() {
            eprintln!("[coordinator] SWA shrink failed: {e}");
        }
        if let Some(next) = next.filter(|_| kv.tokens() == done && done >= MIN_RETAIN) {
            if points.iter().all(|x| x.len < done) {
                let now = self.now();
                points.extend(Point::save(&kv, next, Kind::Prompt, now));
            }
        }
        if points.is_empty() {
            self.release(kv);
            return;
        }
        ids.truncate(done);
        self.retained.push(Retained { kv, hist: ids, points });
    }

    /// Keep each bank within `bank` points on the device: the oldest point of an
    /// overflowing bank goes to RAM (the design's bank overflow). A retained slot
    /// left without points is freed.
    fn enforce_banks(&mut self, active: &mut [Active]) {
        for kind in [Kind::Prompt, Kind::Turn] {
            loop {
                let mut count = 0;
                // (last use, in a running request, owner, point)
                let mut oldest: Option<(u64, bool, usize, usize)> = None;
                let owners = active.iter().map(|a| &a.points).chain(self.retained.iter().map(|r| &r.points));
                for (oi, points) in owners.enumerate() {
                    for (pi, p) in points.iter().enumerate().filter(|(_, p)| p.kind == kind) {
                        count += 1;
                        if oldest.map_or(true, |o| p.last_use < o.0) {
                            oldest = Some((p.last_use, oi < active.len(), oi, pi));
                        }
                    }
                }
                if count <= self.bank {
                    break;
                }
                let (_, running, oi, pi) = oldest.expect("a point to evict");
                if running {
                    let a = &mut active[oi];
                    let p = a.points.swap_remove(pi);
                    Self::store(&mut self.cache, &a.kv, &a.hist, &p);
                } else {
                    let ri = oi - active.len();
                    let r = &mut self.retained[ri];
                    let p = r.points.swap_remove(pi);
                    Self::store(&mut self.cache, &r.kv, &r.hist, &p);
                    if r.points.is_empty() {
                        let r = self.retained.swap_remove(ri);
                        self.release(r.kv);
                    }
                }
            }
        }
    }

    /// Before a decode step: a request about to outgrow its GA reservation grows
    /// now, with room made first (a running request is not failed for pool
    /// pressure while retained slots can be evicted).
    fn grow_active(&mut self, active: &mut [Active]) {
        for a in active.iter_mut() {
            let cap = a.kv.ga_capacity();
            if a.kv.tokens() + 2 * DFLASH_DRAFTS + 2 < cap {
                continue;
            }
            let target = cap + (cap / 8).max(4096);
            if let Err(e) = self.make_room(&a.kv, target).and_then(|_| a.kv.reserve_ga(target)) {
                eprintln!("[coordinator] GA growth to {target} rows failed: {e}");
            }
        }
    }
}

/// Whether prompt `ids` starts with the in-flight prompt `inflight` (worth
/// waiting for: its snapshot will cover `inflight.len()` tokens of `ids`).
fn extends(ids: &[usize], inflight: &[usize]) -> bool {
    inflight.len() >= MIN_RETAIN && ids.len() >= inflight.len() && ids[..inflight.len()] == inflight[..]
}

/// Prefill segments are whole two-lane groups: `forward` runs a prompt as pairs
/// of 4,096-row chunks, so an 8,192-token segment is exactly its own unit. A
/// 4,096-token segment would split into two 2K lanes, which the Sparks run ~18%
/// slower per token (a 1M prompt took 1,421 s against 1,229 s monolithic).
const SEG_QUANTUM: usize = 8192;
/// The longest prefill segment.
const SEG_MAX: usize = 65536;

/// A prefilled request starts decoding with its first token `next`; its
/// prompt-end snapshot joins the device bank (a device copy, no RAM traffic).
fn start(pool: &mut Pool, active: &mut Vec<Active>, p: Prefilling, next: usize, eos: &[usize]) {
    let Prefilling { mut kv, ids, mut points, max, tx, cancel, .. } = p;
    // The prefill is done: its SWA working set goes back (outside the forward).
    if let Err(e) = kv.shrink_swa() {
        eprintln!("[coordinator] SWA shrink failed: {e}");
    }
    let plen = ids.len();
    if plen >= MIN_RETAIN && points.iter().all(|x| x.len != plen) {
        let now = pool.now();
        points.extend(Point::save(&kv, next, Kind::Prompt, now));
    }
    let done = eos.contains(&next) || max <= 1;
    let mut hist = ids;
    hist.push(next);
    let a = Active { kv, last: next, generated: 1, max, tx, cancel, hist, points };
    if a.tx.send(Ok(next)).is_err() || done {
        pool.retire(a);
    } else {
        active.push(a);
    }
}

/// The batching scheduler (perf reset W4): owns the device forward, the wire and
/// the KV slots. Between decode steps it admits queued requests (each prefilled
/// on its own slot with the two-lane prefill), then runs ONE step for every
/// active request: a `decode_batch`, or with the DFlash drafter loaded (perf
/// reset S1) a `spec_step` that drafts 7 tokens per request, verifies them in
/// one target pass and emits the accepted run plus the target's next token.
/// Greedy sampling, as the single-request path.
///
/// KV reuse (perf reset K3, [`Pool`]): every prompt end and completion end of
/// 512+ tokens is kept on the device as a snapshot point inside its slot; a new
/// prompt resumes at its longest exact snapshot, on the device (in place or
/// forked) or restored from RAM, else prefills cold. Only pressure (a bank over
/// its size, no free slot, or not enough GPU memory) copies snapshots to RAM.
///
/// Images (perf reset V2): a prompt's images are encoded when its first segment with image tokens
/// comes up (none when a snapshot already covers them), after `make_room_bytes` has made room for
/// the encoder's transient weights and buffers.
fn scheduler(
    mut fwd: DeviceForward,
    free: Vec<DeviceKv>,
    mut wire: WireClient,
    rx: mpsc::Receiver<Job>,
    vocab: usize,
    eos: Vec<usize>,
    vision: Option<VisionTower>,
) {
    let mut active: Vec<Active> = Vec::new();
    let mut prefilling: std::collections::VecDeque<Prefilling> = std::collections::VecDeque::new();
    let mut deferred: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    // Prefill segment target (perf reset Q1): the longest a running request waits
    // for its next step while prompts prefill.
    let seg_target = std::env::var("MIMO26_PREFILL_SEGMENT_MS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(2000.0)
        / 1000.0;
    let mut sec_per_token = 1.0 / 4000.0;
    let spec = fwd.has_dflash() && std::env::var("MIMO26_SPEC").map(|v| v != "0").unwrap_or(true);
    let cache = match free.first().map(HostCache::from_env) {
        Some(Ok(c)) => c,
        Some(Err(e)) => {
            eprintln!("[hostcache] disabled: {e}");
            None
        }
        None => None,
    };
    let bank = std::env::var("MIMO26_PREFIX_CACHE_ENTRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let mut pool = Pool { free, retained: Vec::new(), cache, clock: 0, bank };
    eprintln!("[coordinator] decode: {}; device snapshot banks {bank} prompt + {bank} turn",
        if spec { "DFlash speculative (block 8)" } else { "one token per step" });
    loop {
        // Admit: block while idle, otherwise take what is queued while slots last
        // (a retained slot counts as available: admission evicts it if needed).
        while !pool.free.is_empty() || !pool.retained.is_empty() {
            // A job deferred behind an identical prefill goes first once that
            // prefill is done (it then resumes from its snapshot).
            let ready = deferred.iter().position(|j: &Job| !prefilling.iter().any(|p| extends(&j.ids, &p.ids)));
            let job = if let Some(i) = ready {
                deferred.remove(i).expect("deferred job")
            } else if active.is_empty() && prefilling.is_empty() {
                match rx.recv() {
                    Ok(j) => j,
                    Err(_) => return, // engine dropped
                }
            } else {
                match rx.try_recv() {
                    Ok(j) => j,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
            };
            if job.cancel.load(Ordering::Relaxed) {
                continue;
            }
            // A prompt that extends one still prefilling waits for it and then
            // forks its snapshot, instead of prefilling the same tokens again (n
            // parallel samples, identical subagent prompts; K3 got this by
            // prefilling during admission).
            if prefilling.iter().any(|p| extends(&job.ids, &p.ids)) {
                deferred.push_back(job);
                continue;
            }
            // Admission (perf reset L1): the request's GA rows must fit now (retained
            // slots are evicted to make room), reserved in one allocation; the prompt
            // resumes at its longest exact snapshot (perf reset K3).
            let rows = admit_rows(job.ids.len(), job.max_tokens);
            let (kv, points, resume) = match pool.admit(&fwd, &job.ids, rows) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("[coordinator] refused a {}-token prompt: {e}", job.ids.len());
                    let _ = job.tx.send(Err(e));
                    continue;
                }
            };
            match resume {
                // An exact snapshot: the first token is known, no forward.
                Some((n, next)) if n == job.ids.len() => {
                    let p = Prefilling { kv, ids: job.ids, done: n, next: Some(next), points, max: job.max_tokens,
                        tx: job.tx, cancel: job.cancel, images: Vec::new(), overlay: None };
                    start(&mut pool, &mut active, p, next, &eos);
                }
                _ => {
                    // The SWA prefill working set now, before any pipelined forward.
                    let mut kv = kv;
                    if let Err(e) = kv.reserve_swa() {
                        eprintln!("[coordinator] refused a {}-token prompt: {e}", job.ids.len());
                        let _ = job.tx.send(Err(e));
                        pool.release(kv);
                        continue;
                    }
                    prefilling.push_back(Prefilling { kv, done: resume.map_or(0, |r| r.0), next: None, ids: job.ids,
                        points, max: job.max_tokens, tx: job.tx, cancel: job.cancel, images: job.images, overlay: None });
                }
            }
            pool.enforce_banks(&mut active);
        }
        // Prefill (perf reset Q1): segments round robin over the prompts in flight
        // until this round has spent about `seg_target`, then one decode step for
        // the running requests. A burst of short prompts prefills in one round
        // (they start decoding together, as before Q1); a long prompt gets one
        // segment per round.
        let round = std::time::Instant::now();
        // Perf reset B1: the short prompts in flight prefill together, one pass over
        // up to BATCH_ROWS rows instead of one pass each (a burst of 16 short prompts
        // queued ~16 per-layer-latency-bound passes for its first tokens).
        if prefilling.len() >= 2 {
            let mut batch: Vec<Prefilling> = Vec::new();
            let mut rows = 0;
            let mut i = 0;
            while i < prefilling.len() {
                let r = prefilling[i].ids.len() - prefilling[i].done;
                if !prefilling[i].cancel.load(Ordering::Relaxed) && prefilling[i].images.is_empty() && rows + r <= BATCH_ROWS {
                    rows += r;
                    batch.push(prefilling.remove(i).expect("prefilling entry"));
                } else {
                    i += 1;
                }
            }
            if batch.len() >= 2 {
                let segs: Vec<Vec<usize>> = batch.iter().map(|p| p.ids[p.done..].to_vec()).collect();
                let seg_refs: Vec<&[usize]> = segs.iter().map(Vec::as_slice).collect();
                let result = {
                    let mut kvs: Vec<&mut DeviceKv> = batch.iter_mut().map(|p| &mut p.kv).collect();
                    fwd.prefill_batch(&seg_refs, &mut kvs, &mut wire)
                };
                match result {
                    Ok(nexts) => {
                        for (mut p, next) in batch.into_iter().zip(nexts) {
                            p.done = p.ids.len();
                            p.next = Some(next);
                            start(&mut pool, &mut active, p, next, &eos);
                        }
                    }
                    Err(e) => {
                        eprintln!("[coordinator] batched prefill of {} prompts failed: {e}", batch.len());
                        for p in batch {
                            let _ = p.tx.send(Err(e.clone()));
                            let mut kv = p.kv;
                            kv.reset();
                            pool.release(kv);
                        }
                    }
                }
            } else {
                for p in batch.into_iter().rev() {
                    prefilling.push_front(p);
                }
            }
        }
        while let Some(mut p) = prefilling.pop_front() {
            if p.cancel.load(Ordering::Relaxed) {
                eprintln!("[coordinator] client gone: parking a prefill at {} of {} tokens", p.done, p.ids.len());
                pool.park(p);
                continue;
            }
            // Alone, a longer segment (fewer pipeline restarts); a new arrival
            // still waits at most about four targets.
            let target = if active.is_empty() && prefilling.is_empty() { 4.0 * seg_target } else { seg_target };
            let len = ((target / sec_per_token) as usize / SEG_QUANTUM * SEG_QUANTUM).clamp(SEG_QUANTUM, SEG_MAX);
            let end = (p.done + len).min(p.ids.len());
            let t0 = std::time::Instant::now();
            // Perf reset V2: the first segment with image tokens encodes the prompt's images.
            if p.overlay.is_none() && p.ids[p.done..end].iter().any(|&id| id >= vocab) {
                match encode_images(&mut pool, vision.as_ref(), &p) {
                    Ok(o) => p.overlay = Some(o),
                    Err(e) => {
                        eprintln!("[coordinator] images of a {}-token prompt: {e}", p.ids.len());
                        let _ = p.tx.send(Err(e));
                        let mut kv = p.kv;
                        kv.reset();
                        let _ = kv.shrink_swa();
                        pool.release(kv);
                        continue;
                    }
                }
            }
            fwd.overlay = p.overlay.take();
            let result = fwd.forward(&p.ids[p.done..end], &mut p.kv, &mut wire);
            p.overlay = fwd.overlay.take();
            match result {
                Ok(l) => {
                    let next = greedy(&l[l.len() - vocab..]);
                    // The rate at this context length sizes the next segment.
                    if end - p.done >= SEG_QUANTUM {
                        sec_per_token = t0.elapsed().as_secs_f64() / (end - p.done) as f64;
                    }
                    p.done = end;
                    p.next = Some(next);
                    if end < p.ids.len() {
                        prefilling.push_back(p);
                    } else {
                        start(&mut pool, &mut active, p, next, &eos);
                    }
                }
                Err(e) => {
                    eprintln!("[coordinator] prefill of a {}-token prompt failed at {}: {e}", p.ids.len(), p.done);
                    let _ = p.tx.send(Err(e));
                    let mut kv = p.kv;
                    kv.reset();
                    let _ = kv.shrink_swa();
                    pool.release(kv);
                }
            }
            // Back to admission and the decode step once the round's budget is spent.
            if round.elapsed().as_secs_f64() >= seg_target {
                break;
            }
        }
        // Bank overflow from the last step's retirements goes to RAM.
        pool.enforce_banks(&mut active);
        // Retire cancelled requests before spending a step on them.
        let mut i = 0;
        while i < active.len() {
            if active[i].cancel.load(Ordering::Relaxed) {
                let a = active.swap_remove(i);
                eprintln!("[coordinator] client gone: ending a request after {} of {} tokens", a.generated, a.max);
                pool.retire(a);
            } else {
                i += 1;
            }
        }
        if active.is_empty() {
            continue;
        }
        pool.grow_active(&mut active);
        if spec {
            let lasts: Vec<usize> = active.iter().map(|a| a.last).collect();
            // Never draft past the token budget: a step emits at most k + 1.
            let ks: Vec<usize> = active.iter().map(|a| (a.max - a.generated - 1).min(DFLASH_DRAFTS)).collect();
            let result = {
                let mut kvs: Vec<&mut DeviceKv> = active.iter_mut().map(|a| &mut a.kv).collect();
                fwd.spec_step(&mut kvs, &lasts, &ks, &mut wire)
            };
            match result {
                Ok(outs) => {
                    let mut keep = Vec::with_capacity(active.len());
                    for (mut a, out) in active.drain(..).zip(outs) {
                        let mut done = false;
                        for next in out {
                            a.last = next;
                            a.generated += 1;
                            a.hist.push(next);
                            done = eos.contains(&next) || a.generated >= a.max;
                            if a.tx.send(Ok(next)).is_err() {
                                done = true;
                            }
                            if done {
                                break;
                            }
                        }
                        if done {
                            pool.retire(a);
                        } else {
                            keep.push(a);
                        }
                    }
                    active = keep;
                }
                Err(e) => {
                    eprintln!("[coordinator] speculative step failed for {} requests: {e}", active.len());
                    for a in active.drain(..) {
                        let _ = a.tx.send(Err(e.clone()));
                        pool.release(a.kv);
                    }
                }
            }
            continue;
        }
        let ids: Vec<usize> = active.iter().map(|a| a.last).collect();
        let result = {
            let mut kvs: Vec<&mut DeviceKv> = active.iter_mut().map(|a| &mut a.kv).collect();
            fwd.decode_batch(&ids, &mut kvs, &mut wire)
        };
        match result {
            Ok(logits) => {
                let mut keep = Vec::with_capacity(active.len());
                for (i, mut a) in active.drain(..).enumerate() {
                    let next = greedy(&logits[i * vocab..(i + 1) * vocab]);
                    a.last = next;
                    a.generated += 1;
                    a.hist.push(next);
                    let done = eos.contains(&next) || a.generated >= a.max;
                    if a.tx.send(Ok(next)).is_err() || done {
                        pool.retire(a);
                    } else {
                        keep.push(a);
                    }
                }
                active = keep;
            }
            Err(e) => {
                eprintln!("[coordinator] decode step failed for {} requests: {e}", active.len());
                for a in active.drain(..) {
                    let _ = a.tx.send(Err(e.clone()));
                    pool.release(a.kv);
                }
            }
        }
    }
}

impl CoordinatorEngine {
    pub fn new(
        cfg: Config,
        model: ServingModel,
        tok: BpeTokenizer,
        wire: WireClient,
    ) -> Self {
        let caches = (0..cfg.num_hidden_layers).map(|l| Fp8KvCache::for_layer(&cfg, l)).collect();
        Self { cfg, backend: Backend::Host { model, caches: Mutex::new(caches), wire: Mutex::new(wire) }, tok, max_context: None,
            vision_tokens: None }
    }

    /// The device-resident engine (perf reset R1) behind the batching scheduler
    /// (W4). `kv` is the first slot; `MIMO26_MAX_SLOTS` (default 8) sets how many
    /// requests decode together, each further slot starting at
    /// `MIMO26_SLOT_KV_TOKENS` (default 8192) GA rows and growing on demand.
    pub fn new_device(cfg: Config, mut fwd: DeviceForward, kv: DeviceKv, tok: BpeTokenizer, wire: WireClient) -> Self {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let slots = env("MIMO26_MAX_SLOTS", 8).clamp(1, 64);
        let slot_kv = env("MIMO26_SLOT_KV_TOKENS", 4096);
        // Perf reset V2: the image encoder, in page-locked RAM until an image comes (`MIMO26_VISION=0`: off).
        let vision = if std::env::var("MIMO26_VISION").map(|v| v == "0").unwrap_or(false) {
            None
        } else {
            let t = std::time::Instant::now();
            match VisionTower::load_dir(&crate::load::weights_dir()) {
                Ok(v) => {
                    eprintln!("[vision] image encoder: {:.2} GB page-locked in {:.1} s", v.weight_bytes() as f64 / 1e9,
                        t.elapsed().as_secs_f64());
                    Some(v)
                }
                Err(e) => {
                    eprintln!("[vision] image encoder not loaded: {e}");
                    None
                }
            }
        };
        let vision_tokens = vision.as_ref().map(|v| (v.vision_start, v.vision_end));
        let mut free = vec![kv];
        for _ in 1..slots {
            free.push(fwd.new_kv(slot_kv).expect("scheduler KV slot"));
        }
        // Every per-step buffer at its steady-state size before measuring what
        // is free, so the maximum context and admission see the real room.
        fwd.warm_scratch(slots).expect("scheduler scratch");
        // Max context (perf reset L1): what one slot can grow to with the pool idle.
        let max_context = device_free_bytes().ok().map(|f| {
            let kv = &free[0];
            let room = f.saturating_sub(kv.swa_prefill_bytes() + KV_MARGIN_BYTES) / kv.ga_token_bytes();
            (kv.ga_capacity() + room).saturating_sub(8192 + 64).min(1 << 20)
        });
        eprintln!("[coordinator] batching scheduler: {slots} slots ({slot_kv} GA rows each to start); max context \
            {max_context:?} tokens per request (GPU {:.2} GiB free)", device_free_bytes().unwrap_or(0) as f64 / (1u64 << 30) as f64);
        let (tx, rx) = mpsc::channel();
        let (vocab, eos) = (cfg.vocab_size, cfg.eos_token_ids.iter().map(|&x| x as usize).collect());
        std::thread::Builder::new()
            .name("mimo26-scheduler".into())
            .spawn(move || scheduler(fwd, free, wire, rx, vocab, eos, vision))
            .expect("spawn scheduler");
        Self { cfg, backend: Backend::Device { jobs: Mutex::new(tx) }, tok, max_context, vision_tokens }
    }

    /// Host reference path: reset the request state, then run one forward step;
    /// returns the last row's logits. `fresh` is true for a request's prefill.
    fn step(&self, st: &mut StepState<'_>, ids: &[usize], fresh: bool) -> Result<Vec<f32>, String> {
        let vocab = self.cfg.vocab_size;
        let StepState { model, caches, wire } = st;
        if fresh {
            for c in caches.iter_mut() {
                c.clear();
            }
        }
        let (logits, _) = model.forward(ids, caches, Some(&mut **wire));
        Ok(logits[logits.len() - vocab..].to_vec())
    }
}

/// Locked per-request state of the host reference path.
struct StepState<'a> {
    model: &'a ServingModel,
    caches: std::sync::MutexGuard<'a, Vec<Fp8KvCache>>,
    wire: std::sync::MutexGuard<'a, WireClient>,
}

thread_local! {
    /// The prompt this connection's thread last tokenized to count it (perf reset
    /// Q3): the API counts a prompt's tokens and then generates it on the same
    /// thread, so generation reuses the ids instead of encoding again (about
    /// 0.35 s at 512K tokens).
    static LAST_ENCODE: std::cell::RefCell<Option<(String, Vec<u32>)>> = const { std::cell::RefCell::new(None) };
}

impl CoordinatorEngine {
    /// `prompt`'s token ids, from [`LAST_ENCODE`] when it is the prompt just counted.
    fn encode_prompt(&self, prompt: &str) -> Vec<u32> {
        let hit = LAST_ENCODE.with(|c| c.borrow_mut().take().filter(|(p, _)| p == prompt).map(|(_, ids)| ids));
        hit.unwrap_or_else(|| encode_marked(&self.tok, self.vision_tokens, prompt))
    }
}

impl Engine for CoordinatorEngine {
    fn vision(&self) -> bool {
        self.vision_tokens.is_some()
    }

    fn max_context(&self) -> Option<usize> {
        self.max_context
    }

    fn tokenize(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> usize {
        let rendered = self.render_chat(messages, tools, thinking);
        let ids = encode_marked(&self.tok, self.vision_tokens, &rendered);
        let n = ids.len();
        LAST_ENCODE.with(|c| *c.borrow_mut() = Some((rendered, ids)));
        n
    }

    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> String {
        let msgs = map_messages(messages);
        // D2b item 1/2: the tools block serialises the client's WHOLE tool
        // object (kept verbatim in `Tool.raw`, in the client's key order and
        // including `description`) with the template's `tojson`.
        let tool_json: Vec<mimo26_api::json::Json> = tools.iter().map(|t| t.raw.clone()).collect();
        let opts = crate::chat::ChatOptions {
            add_generation_prompt: true,
            enable_thinking: thinking,
        };
        crate::chat::render_chat(&msgs, &tool_json, &opts)
    }

    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String> {
        let prompt_ids: Vec<usize> = self.encode_prompt(prompt).iter().map(|&x| x as usize).collect();
        let cancel = params.cancel.clone().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let vocab = self.cfg.vocab_size;
        let eos: Vec<usize> = self.cfg.eos_token_ids.iter().map(|&x| x as usize).collect();

        // Where tokens come from: the batching scheduler (device), or the host
        // reference path under its locks. D5: a poisoned guard is recovered
        // instead of failing every later request.
        enum Source<'a> {
            Sched { rx: mpsc::Receiver<Result<usize, String>>, _cancel: CancelOnDrop },
            Host(StepState<'a>),
        }
        let max = params.max_tokens.min(65536).max(1);
        let mut src = match &self.backend {
            Backend::Device { jobs } => {
                let (tx, rx) = mpsc::channel();
                jobs.lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .send(Job { ids: prompt_ids.clone(), max_tokens: max, tx, cancel: cancel.clone(),
                        images: image_spans(&prompt_ids, &params.images)? })
                    .map_err(|_| "scheduler stopped".to_string())?;
                Source::Sched { rx, _cancel: CancelOnDrop(cancel.clone()) }
            }
            Backend::Host { model, caches, wire } => Source::Host(StepState {
                model,
                caches: caches.lock().unwrap_or_else(|p| p.into_inner()),
                wire: wire.lock().unwrap_or_else(|p| p.into_inner()),
            }),
        };
        // D1: fresh KV cache per request (the scheduler resets its slot; the host
        // path clears its caches on the prefill step).
        // Perf reset Q2: while a token is slow to come (a long prefill), an empty
        // delta every 15 s keeps the client's connection alive, and a client that
        // left (the API sets `cancel` on a failed write) ends the request, which
        // frees its slot and keeps its prefill as a snapshot for a retry.
        let next_token = |src: &mut Source<'_>, ids: &[usize], fresh: bool, on_delta: &mut dyn FnMut(&str)|
            -> Result<usize, String> {
            match src {
                Source::Sched { rx, .. } => loop {
                    match rx.recv_timeout(std::time::Duration::from_secs(15)) {
                        Ok(r) => break r,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            emit_delta(on_delta, "")?;
                            if cancel.load(Ordering::Relaxed) {
                                return Err("client gone".to_string());
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => return Err("scheduler stopped".to_string()),
                    }
                },
                Source::Host(st) => {
                    let logits = self.step(st, ids, fresh)?;
                    debug_assert_eq!(logits.len(), vocab);
                    Ok(greedy(&logits))
                }
            }
        };
        let mut next = next_token(&mut src, &prompt_ids, true, on_delta)?;
        let mut gen_ids: Vec<usize> = vec![next];

        // Decode, streaming one delta per sampled token (the API forwards each as
        // an SSE content delta). `pending` holds token ids whose text could be a
        // stop prefix, so a stop sequence is never emitted early.
        let mut full_text = String::new();
        let mut pending: Vec<u32> = Vec::new();
        let mut finish_reason = "length".to_string();

        loop {
            if eos.contains(&next) {
                finish_reason = "stop".to_string();
                break;
            }
            pending.push(next as u32);
            let (emit, stopped) = flush_pending(&self.tok, &params.stop, &mut pending);
            if !emit.is_empty() {
                emit_delta(on_delta, &emit)?;
                full_text.push_str(&emit);
            }
            if stopped {
                finish_reason = "stop".to_string();
                break;
            }
            if gen_ids.len() >= max {
                break; // length
            }
            if cancel.load(Ordering::Relaxed) {
                return Err("client gone".to_string());
            }
            next = next_token(&mut src, &[next], false, on_delta)?;
            gen_ids.push(next);
        }

        // Flush any held-back tail (a partial stop that never completed, or the
        // last token before EOS/length).
        let tail = self.tok.decode(&pending);
        if !tail.is_empty() {
            emit_delta(on_delta, &tail)?;
            full_text.push_str(&tail);
        }

        Ok(GenerateOutcome {
            text: full_text,
            finish_reason,
            completion_tokens: gen_ids.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimo26_api::types::{Tool, ToolCall, ToolFunction};

    /// D2: assistant tool calls in the message history must be rendered (not
    /// dropped), so the model sees them per the checkpoint template.
    #[test]
    fn tool_calls_are_rendered_not_dropped() {
        let tools = vec![Tool {
            r#type: "function".into(),
            function: ToolFunction {
                name: "read_file".into(),
                description: None,
                parameters: Some(
                    mimo26_api::json::parse(r#"{"properties":{"path":{"type":"string"}}}"#).unwrap(),
                ),
            },
            raw: mimo26_api::json::parse(
                r#"{"type":"function","function":{"name":"read_file","parameters":{"properties":{"path":{"type":"string"}}}}}"#,
            )
            .unwrap(),
        }];
        let msgs = vec![ChatMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                r#type: "function".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"/tmp/x"}"#.into(),
            }],
        }];
        let mapped = map_messages(&msgs);
        let tool_json: Vec<mimo26_api::json::Json> = tools.iter().map(|t| t.raw.clone()).collect();
        let rendered =
            crate::chat::render_chat(&mapped, &tool_json, &crate::chat::ChatOptions::default());
        assert!(rendered.contains("read_file"), "tool name dropped: {rendered}");
        assert!(rendered.contains("parameter=path"), "param dropped: {rendered}");
        assert!(rendered.contains("/tmp/x"), "value dropped: {rendered}");
        // The tools block (system-side schemas) must be present, not just the calls.
        assert!(rendered.contains("You are provided with the following tools"), "tools block dropped: {rendered}");
        assert!(rendered.contains("tools"), "tools tag dropped: {rendered}");
        assert!(rendered.contains("\"properties\""), "schema dropped: {rendered}");
    }

    /// D2b item 3: history tool-call arguments render in the CLIENT's JSON
    /// order, never the tool schema's declared order (vLLM json.loads the
    /// arguments string before templating).
    #[test]
    fn tool_call_arguments_render_in_client_order() {
        let tc = ToolCall {
            id: "call_1".into(),
            r#type: "function".into(),
            name: "todo_write".into(),
            // `status` before `content`, the reverse of the schema's declared
            // `properties` order.
            arguments: r#"{"status": "in_progress", "content": "fix typo"}"#.into(),
        };
        let mapped = map_tool_call(&tc);
        let keys: Vec<&str> = mapped.arguments.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["status", "content"]);
    }

    /// D5: a panicking on_delta must be caught (return Err), not unwind through
    /// the held engine locks.
    #[test]
    fn emit_catches_a_panicking_callback() {
        let r = emit_delta(&mut |_| panic!("boom"), "x");
        assert_eq!(r, Err("on_delta panicked".to_string()));
        // A non-panicking callback succeeds.
        let r = emit_delta(&mut |_| {}, "ok");
        assert!(r.is_ok());
    }

    /// D5: a poisoned mutex guard must be recoverable via into_inner.
    #[test]
    fn poisoned_guard_is_recovered() {
        let m = std::sync::Mutex::new(0i32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = m.lock().unwrap();
            panic!("poison");
        }));
        assert!(m.is_poisoned());
        let g = m.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!(*g, 0);
    }
}

#[cfg(test)]
mod vision_tests {
    use super::{image_spans, image_token_id};
    use mimo26_api::engine::ImageInput;
    use std::sync::Arc;

    fn img(hash: u64, tokens: usize) -> Arc<ImageInput> {
        Arc::new(ImageInput { hash, tokens, width: 64, height: 64, rgb: Vec::new() })
    }

    #[test]
    fn image_token_ids_are_past_the_vocabulary_and_follow_the_image() {
        let a: Vec<u32> = (0..64).map(|i| image_token_id(7, i)).collect();
        let b: Vec<u32> = (0..64).map(|i| image_token_id(8, i)).collect();
        assert!(a.iter().chain(&b).all(|&t| t >= 0x8000_0000));
        assert_ne!(a, b, "different images, different ids");
        assert_eq!(a, (0..64).map(|i| image_token_id(7, i)).collect::<Vec<_>>(), "same image, same ids");
    }

    #[test]
    fn image_spans_locate_each_image_in_order() {
        let (x, y) = (img(1, 4), img(2, 3));
        let mut ids: Vec<usize> = vec![10, 11, 151652];
        ids.extend((0..4).map(|i| image_token_id(1, i) as usize));
        ids.extend([151653, 12, 151652]);
        ids.extend((0..3).map(|i| image_token_id(2, i) as usize));
        ids.extend([151653, 13]);
        let spans = image_spans(&ids, &[x.clone(), y.clone()]).unwrap();
        assert_eq!(spans.iter().map(|s| s.start).collect::<Vec<_>>(), vec![3, 10]);
        // Order, count and length mismatches are errors, never a silent misplacement.
        assert!(image_spans(&ids, &[y.clone(), x.clone()]).is_err());
        assert!(image_spans(&ids, &[x.clone()]).is_err());
        assert!(image_spans(&ids, &[x, img(2, 4)]).is_err());
    }
}
