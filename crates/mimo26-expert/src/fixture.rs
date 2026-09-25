//! The real-block golden fixture — `bench/fixtures/expert_nibble_fixture.json`.
//!
//! # What the fixture is
//!
//! 27 **real** expert blocks from the local checkpoint copy (READ-ONLY), produced
//! by `harness/expert_fixture.py`: layers {1, 24, 46} x experts {0, 7, 255} x
//! projections {gate, up, down}. Per block:
//!
//! * `weight_sha256` / `scale_sha256` — the raw bytes' sha256, so the harness
//!   can prove it is looking at the same bytes the reference unpacked;
//! * `positions` — 2,048 deterministic `(row, col)` samples;
//! * `expected_f32_bits` — the reference `spike/mxfp4.unpack` f32 **bit
//!   patterns** at those positions (hex, 8 chars);
//! * scale-byte statistics (`scale_byte_min/max`, `scale_byte_255_count`,
//!   `saturated_count`).
//!
//! The fixture is sha-pinned by `expert_nibble_fixture.json.sha256`.
//!
//! # Why bit patterns, not values
//!
//! The unpack path is exact: every E2M1 value times an exact power of two is
//! representable in f32 (except at the saturation boundary), so the reference
//! and the kernel must agree **bitwise** — including `-0.0` (nibble 8) vs
//! `0.0` (nibble 0), which compare equal as values. A tolerance-based check
//! would pass a nibble-swapped implementation on any block whose even and odd
//! nibbles happen to decode to the same magnitude; the bitwise check cannot.
//!
//! # Finding recorded by the fixture (23 Sep 2026 AEST)
//!
//! `scale_byte_max` is 121-125 and `scale_byte_255_count` is 0 across all 27
//! real blocks: **the reserved 255 clamp does not occur in real expert data**.
//! The T10 detector therefore stays synthetic (`tests/t10_scale.rs`), and the
//! real-block check pins the *mapping* (`2^(b-127)`) rather than the clamp.
//! `saturated_count` is 0 too, so the f32 saturation is likewise synthetic-only
//! on real data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::mxfp4;
use crate::slice::Proj;
use crate::{ExpertError, NaiveBits};

/// One block of the fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureBlock {
    /// `model.layers.L.mlp.experts.E.<proj>`.
    pub name: String,
    /// Logical `[out, in]`.
    pub shape: [usize; 2],
    /// sha256 of the packed payload bytes.
    pub weight_sha256: String,
    /// sha256 of the scale bytes.
    pub scale_sha256: String,
    /// Sampled `(row, col)` positions.
    pub positions: Vec<(usize, usize)>,
    /// Expected f32 bit patterns at those positions.
    pub expected_f32_bits: Vec<u32>,
    /// Minimum scale byte in the block.
    pub scale_byte_min: u8,
    /// Maximum scale byte in the block.
    pub scale_byte_max: u8,
    /// How many scale bytes are the reserved 255.
    pub scale_byte_255_count: usize,
    /// How many reference values hit the f32 saturation clamp.
    pub saturated_count: usize,
}

impl FixtureBlock {
    /// The layer id parsed out of the name.
    pub fn layer(&self) -> usize {
        self.name
            .split('.')
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX)
    }

    /// The expert id parsed out of the name.
    pub fn expert(&self) -> usize {
        self.name
            .split('.')
            .nth(5)
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX)
    }

    /// The projection parsed out of the name.
    pub fn proj(&self) -> Option<Proj> {
        let last = self.name.rsplit('.').next()?;
        match last {
            "gate_proj" => Some(Proj::Gate),
            "up_proj" => Some(Proj::Up),
            "down_proj" => Some(Proj::Down),
            _ => None,
        }
    }

    /// The expected value at sample `i`, as f32.
    pub fn expected_f32(&self, i: usize) -> f32 {
        f32::from_bits(self.expected_f32_bits[i])
    }
}

/// The whole fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixture {
    /// E8M0 block width (32).
    pub block: usize,
    /// The blocks.
    pub blocks: Vec<FixtureBlock>,
    /// The layers the fixture covers.
    pub layers: Vec<usize>,
    /// The experts the fixture covers.
    pub experts: Vec<usize>,
    /// The projections the fixture covers.
    pub projections: Vec<String>,
    /// The reference description string.
    pub reference: String,
    /// The source description string.
    pub source: String,
}

impl Fixture {
    /// The block with this name.
    pub fn block(&self, name: &str) -> Option<&FixtureBlock> {
        self.blocks.iter().find(|b| b.name == name)
    }

    /// Blocks for one projection.
    pub fn blocks_for(&self, p: Proj) -> Vec<&FixtureBlock> {
        self.blocks.iter().filter(|b| b.proj() == Some(p)).collect()
    }

    /// Total sampled positions across all blocks.
    pub fn total_samples(&self) -> usize {
        self.blocks.iter().map(|b| b.positions.len()).sum()
    }
}

/// The fixture's default path, relative to the repo root.
pub const FIXTURE_REL: &str = "bench/fixtures/expert_nibble_fixture.json";

/// The repo root, from this crate's manifest dir.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The fixture path (repo-relative by default; `MIMO26_EXPERT_FIXTURE` overrides).
pub fn fixture_path() -> PathBuf {
    match std::env::var("MIMO26_EXPERT_FIXTURE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => repo_root().join(FIXTURE_REL),
    }
}

/// Load the fixture from its default path.
pub fn load() -> Result<Fixture, ExpertError> {
    load_from(&fixture_path())
}

/// Load the fixture from `path`.
pub fn load_from(path: &Path) -> Result<Fixture, ExpertError> {
    let text = std::fs::read_to_string(path).map_err(|e| ExpertError::Fixture {
        what: format!("cannot read {}: {e}", path.display()),
    })?;
    parse(&text)
}

/// Parse the fixture JSON.
///
/// Hand-rolled: the workspace convention is zero dependencies (`mimo26-load`,
/// `mimo26-wire`, `mimo26-lanesim`, `mimo26-repack` are all std-only), and the
/// fixture's shape is fixed and simple. The parser is strict — an unexpected
/// key or a malformed value is a loud error, never a silent default.
pub fn parse(text: &str) -> Result<Fixture, ExpertError> {
    let v = JsonParser::new(text).parse()?;
    let obj = v.as_object().ok_or_else(|| ExpertError::Fixture {
        what: "top level is not an object".into(),
    })?;

    let block = obj
        .get("block")
        .and_then(|v| v.as_usize())
        .ok_or_else(|| ExpertError::Fixture { what: "missing `block`".into() })?;
    let blocks_v = obj
        .get("blocks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ExpertError::Fixture { what: "missing `blocks`".into() })?;

    let mut blocks = Vec::with_capacity(blocks_v.len());
    for (i, bv) in blocks_v.iter().enumerate() {
        let b = bv.as_object().ok_or_else(|| ExpertError::Fixture {
            what: format!("blocks[{i}] is not an object"),
        })?;
        let name = b
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ExpertError::Fixture { what: format!("blocks[{i}].name") })?
            .to_string();
        let shape_v = b
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExpertError::Fixture { what: format!("blocks[{i}].shape") })?;
        if shape_v.len() != 2 {
            return Err(ExpertError::Fixture {
                what: format!("blocks[{i}].shape has {} dims, expected 2", shape_v.len()),
            });
        }
        let shape = [
            shape_v[0].as_usize().ok_or_else(|| ExpertError::Fixture {
                what: format!("blocks[{i}].shape[0]"),
            })?,
            shape_v[1].as_usize().ok_or_else(|| ExpertError::Fixture {
                what: format!("blocks[{i}].shape[1]"),
            })?,
        ];
        let positions_v = b
            .get("positions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExpertError::Fixture { what: format!("blocks[{i}].positions") })?;
        let mut positions = Vec::with_capacity(positions_v.len());
        for (j, pv) in positions_v.iter().enumerate() {
            let pa = pv.as_array().ok_or_else(|| ExpertError::Fixture {
                what: format!("blocks[{i}].positions[{j}]"),
            })?;
            if pa.len() != 2 {
                return Err(ExpertError::Fixture {
                    what: format!("blocks[{i}].positions[{j}] has {} dims", pa.len()),
                });
            }
            positions.push((
                pa[0].as_usize().ok_or_else(|| ExpertError::Fixture {
                    what: format!("blocks[{i}].positions[{j}][0]"),
                })?,
                pa[1].as_usize().ok_or_else(|| ExpertError::Fixture {
                    what: format!("blocks[{i}].positions[{j}][1]"),
                })?,
            ));
        }
        let bits_v = b
            .get("expected_f32_bits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExpertError::Fixture {
                what: format!("blocks[{i}].expected_f32_bits"),
            })?;
        let mut expected_f32_bits = Vec::with_capacity(bits_v.len());
        for (j, hv) in bits_v.iter().enumerate() {
            let s = hv.as_str().ok_or_else(|| ExpertError::Fixture {
                what: format!("blocks[{i}].expected_f32_bits[{j}] is not a string"),
            })?;
            let bits = u32::from_str_radix(s, 16).map_err(|_| ExpertError::Fixture {
                what: format!("blocks[{i}].expected_f32_bits[{j}] = {s:?} is not hex"),
            })?;
            expected_f32_bits.push(bits);
        }
        if positions.len() != expected_f32_bits.len() {
            return Err(ExpertError::Fixture {
                what: format!(
                    "blocks[{i}] has {} positions but {} expected bits",
                    positions.len(),
                    expected_f32_bits.len()
                ),
            });
        }
        let get_str = |k: &str| -> Result<String, ExpertError> {
            b.get(k)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| ExpertError::Fixture {
                    what: format!("blocks[{i}].{k}"),
                })
        };
        let get_usize = |k: &str| -> Result<usize, ExpertError> {
            b.get(k)
                .and_then(|v| v.as_usize())
                .ok_or_else(|| ExpertError::Fixture {
                    what: format!("blocks[{i}].{k}"),
                })
        };
        blocks.push(FixtureBlock {
            name,
            shape,
            weight_sha256: get_str("weight_sha256")?,
            scale_sha256: get_str("scale_sha256")?,
            positions,
            expected_f32_bits,
            scale_byte_min: get_usize("scale_byte_min")? as u8,
            scale_byte_max: get_usize("scale_byte_max")? as u8,
            scale_byte_255_count: get_usize("scale_byte_255_count")?,
            saturated_count: get_usize("saturated_count")?,
        });
    }

    let get_usize_arr = |k: &str| -> Result<Vec<usize>, ExpertError> {
        obj.get(k)
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExpertError::Fixture { what: format!("missing `{k}`") })?
            .iter()
            .map(|v| {
                v.as_usize()
                    .ok_or_else(|| ExpertError::Fixture { what: format!("`{k}` element") })
            })
            .collect()
    };
    let get_str_arr = |k: &str| -> Result<Vec<String>, ExpertError> {
        obj.get(k)
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExpertError::Fixture { what: format!("missing `{k}`") })?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| ExpertError::Fixture { what: format!("`{k}` element") })
            })
            .collect()
    };

    Ok(Fixture {
        block,
        blocks,
        layers: get_usize_arr("layers")?,
        experts: get_usize_arr("experts")?,
        projections: get_str_arr("projections")?,
        reference: obj
            .get("reference")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        source: obj
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    })
}

/// Check the fixture's sampled positions against a **synthetic** block built
/// from the same bytes.
///
/// The real fixture carries only the sampled bit patterns, not the 33 MB of
/// payload (deliberately — see `harness/expert_fixture.py`). The GPU cell
/// re-reads the real bytes from the checkpoint and compares; this function is
/// the CPU-side half: given a block's payload+scales, it recomputes the
/// expected bits and reports the first mismatch.
///
/// Returns `Ok(())` when every sampled position matches bitwise.
pub fn check_block(
    block: &FixtureBlock,
    payload: &[u8],
    scales: &[u8],
    naive: NaiveBits,
) -> Result<(), ExpertError> {
    let in_cols = block.shape[1];
    let half = in_cols / 2;
    let srow = in_cols / mxfp4::BLOCK;
    if payload.len() != block.shape[0] * half {
        return Err(ExpertError::ShapeMismatch {
            what: format!(
                "{}: payload {} B, expected {}",
                block.name,
                payload.len(),
                block.shape[0] * half
            ),
        });
    }
    if scales.len() != block.shape[0] * srow {
        return Err(ExpertError::ShapeMismatch {
            what: format!(
                "{}: scales {} B, expected {}",
                block.name,
                scales.len(),
                block.shape[0] * srow
            ),
        });
    }
    for (i, &(r, c)) in block.positions.iter().enumerate() {
        let prow = &payload[r * half..(r + 1) * half];
        let srowb = &scales[r * srow..(r + 1) * srow];
        let got = mxfp4::unpack_element(prow, srowb, c, naive);
        let want = block.expected_f32_bits[i];
        if got.to_bits() != want {
            return Err(ExpertError::Fixture {
                what: format!(
                    "{}: sample {i} at (row {r}, col {c}): got {:#010x} ({got:?}), want {want:#010x} ({:?})",
                    block.name,
                    got.to_bits(),
                    f32::from_bits(want)
                ),
            });
        }
    }
    Ok(())
}

/// Count how many sampled positions a block's bytes match bitwise.
///
/// The negative tests use this: a wrong implementation must match **fewer**
/// than all of them, and the count is the evidence.
pub fn count_matches(
    block: &FixtureBlock,
    payload: &[u8],
    scales: &[u8],
    naive: NaiveBits,
) -> usize {
    let in_cols = block.shape[1];
    let half = in_cols / 2;
    let srow = in_cols / mxfp4::BLOCK;
    let mut n = 0;
    for (i, &(r, c)) in block.positions.iter().enumerate() {
        let prow = &payload[r * half..(r + 1) * half];
        let srowb = &scales[r * srow..(r + 1) * srow];
        let got = mxfp4::unpack_element(prow, srowb, c, naive);
        if got.to_bits() == block.expected_f32_bits[i] {
            n += 1;
        }
    }
    n
}

// ---------------------------------------------------------------------------
// minimal strict JSON (std-only; the workspace convention is zero deps)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }
    fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self { b: s.as_bytes(), i: 0 }
    }

    fn err<T>(&self, what: &str) -> Result<T, ExpertError> {
        Err(ExpertError::Fixture {
            what: format!("json at byte {}: {what}", self.i),
        })
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn parse(&mut self) -> Result<Json, ExpertError> {
        let v = self.value()?;
        self.ws();
        if self.i != self.b.len() {
            return self.err("trailing bytes after the top-level value");
        }
        Ok(v)
    }

    fn value(&mut self) -> Result<Json, ExpertError> {
        self.ws();
        if self.i >= self.b.len() {
            return self.err("unexpected end of input");
        }
        match self.b[self.i] {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'n' => self.lit("null", Json::Null),
            _ => self.number(),
        }
    }

    fn lit(&mut self, s: &str, v: Json) -> Result<Json, ExpertError> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            self.err("bad literal")
        }
    }

    fn object(&mut self) -> Result<Json, ExpertError> {
        self.i += 1; // '{'
        let mut m = BTreeMap::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.i >= self.b.len() || self.b[self.i] != b':' {
                return self.err("expected ':'");
            }
            self.i += 1;
            let v = self.value()?;
            m.insert(k, v);
            self.ws();
            if self.i >= self.b.len() {
                return self.err("unterminated object");
            }
            match self.b[self.i] {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self) -> Result<Json, ExpertError> {
        self.i += 1; // '['
        let mut a = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            return Ok(Json::Arr(a));
        }
        loop {
            let v = self.value()?;
            a.push(v);
            self.ws();
            if self.i >= self.b.len() {
                return self.err("unterminated array");
            }
            match self.b[self.i] {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn string(&mut self) -> Result<String, ExpertError> {
        if self.i >= self.b.len() || self.b[self.i] != b'"' {
            return self.err("expected a string");
        }
        self.i += 1;
        let mut s = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(s);
                }
                b'\\' => {
                    self.i += 1;
                    if self.i >= self.b.len() {
                        return self.err("unterminated escape");
                    }
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            if self.i + 4 > self.b.len() {
                                return self.err("short \\u escape");
                            }
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
                                .map_err(|_| ExpertError::Fixture {
                                    what: "bad \\u escape".into(),
                                })?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| {
                                ExpertError::Fixture { what: "bad \\u escape".into() }
                            })?;
                            self.i += 4;
                            s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return self.err("unknown escape"),
                    }
                }
                _ => {
                    // UTF-8 passthrough: copy the whole char.
                    let start = self.i;
                    let len = utf8_len(c);
                    self.i += len;
                    if self.i > self.b.len() {
                        return self.err("truncated UTF-8");
                    }
                    s.push_str(
                        std::str::from_utf8(&self.b[start..self.i]).map_err(|_| {
                            ExpertError::Fixture { what: "invalid UTF-8".into() }
                        })?,
                    );
                }
            }
        }
        self.err("unterminated string")
    }

    fn number(&mut self) -> Result<Json, ExpertError> {
        let start = self.i;
        if self.i < self.b.len() && (self.b[self.i] == b'-' || self.b[self.i] == b'+') {
            self.i += 1;
        }
        while self.i < self.b.len()
            && (self.b[self.i].is_ascii_digit()
                || matches!(self.b[self.i], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| ExpertError::Fixture {
            what: "bad number".into(),
        })?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| ExpertError::Fixture { what: format!("bad number {s:?}") })
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}
