//! MiMo tokenizer (A2 F1 "tokenizer encode and decode") — a GPT-2 ByteLevel
//! BPE, twin of the served `tokenizer.json` (vocab 151,643 + 151,387 merges +
//! 32 added tokens). Golden-locked against HF `tokenizers` encode/decode for the
//! X1a needle and chat-template strings.
//!
//! Scope note: the served normalizer is NFC, a no-op for the ASCII surface the
//! X1a gate exercises; non-ASCII NFC normalization is not implemented (recorded
//! limitation). The GPT-2 pre-tokenizer regex is hand-rolled over Rust's
//! `char::is_alphabetic`/`is_numeric`/`is_whitespace` (exact for ASCII; the
//! `\p{L}`/`\p{N}` Unicode boundary is approximate for non-ASCII).

use std::collections::HashMap;

/// GPT-2 `bytes_to_unicode()`: byte b -> the unicode char used in the vocab
/// (space 0x20 -> 'Ġ' U+0120; printable ASCII and most of Latin-1 map to
/// themselves; the rest map to 256+n in ascending byte order).
pub fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let mapped = if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
            b
        } else {
            let v = 256 + n;
            n += 1;
            v
        };
        map[b as usize] = char::from_u32(mapped).expect("valid unicode");
    }
    map
}

/// Inverse of [`bytes_to_unicode`] for decode.
pub fn unicode_to_bytes() -> HashMap<char, u8> {
    let mut m = HashMap::new();
    for (b, c) in bytes_to_unicode().iter().enumerate() {
        m.insert(*c, b as u8);
    }
    m
}

/// One added token (matched whole, before BPE). `special` tokens decode to ''.
#[derive(Debug, Clone)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub special: bool,
}

/// A GPT-2 ByteLevel BPE tokenizer over an in-memory vocab + merges.
#[derive(Debug, Clone, Default)]
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merges: HashMap<(String, String), u32>,
    added_tokens: Vec<AddedToken>,
}

impl BpeTokenizer {
    /// Build from raw parts (unit tests).
    pub fn from_parts(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        added_tokens: Vec<AddedToken>,
    ) -> Self {
        let id_to_token = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let merges_map = merges.into_iter().enumerate().map(|(i, p)| (p, i as u32)).collect();
        BpeTokenizer { vocab, id_to_token, merges: merges_map, added_tokens }
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
    pub fn merges_len(&self) -> usize {
        self.merges.len()
    }

    /// Longest added-token prefix of `text`.
    fn match_added(&self, text: &str) -> Option<&AddedToken> {
        self.added_tokens.iter().filter(|t| text.starts_with(&t.content))
            .max_by_key(|t| t.content.len())
    }

    /// Encode text to token ids (NFC is a no-op for the ASCII surface served).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut pos = 0usize;
        let mut pending = String::new();
        while pos < text.len() {
            let rest = &text[pos..];
            if let Some(tok) = self.match_added(rest) {
                if !pending.is_empty() {
                    self.encode_segment(&pending, &mut ids);
                    pending.clear();
                }
                ids.push(tok.id);
                pos += tok.content.len();
            } else {
                let c = rest.chars().next().expect("non-empty");
                pending.push(c);
                pos += c.len_utf8();
            }
        }
        if !pending.is_empty() {
            self.encode_segment(&pending, &mut ids);
        }
        ids
    }

    fn encode_segment(&self, segment: &str, ids: &mut Vec<u32>) {
        for raw in gpt2_pretokenize(segment) {
            let word = byte_encode(&raw);
            for token in bpe_merge(&word, &self.merges) {
                let id = *self.vocab.get(&token).unwrap_or_else(|| {
                    panic!("byte-level token not in vocab: {token:?} (raw {raw:?}, word {word:?})")
                });
                ids.push(id);
            }
        }
    }

    /// Decode ids to text (special tokens decode to ''; the rest byte-decode).
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Decode ids to raw bytes, without the lossy UTF-8 replacement (D7). Used by
    /// the streaming holdback to detect a character split across byte-level BPE
    /// tokens before it would emit U+FFFD.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let byte_map = unicode_to_bytes();
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            // Added tokens: special decode to '' (HF skip_special_tokens default),
            // non-special decode to their content.
            if let Some(tok) = self.added_tokens.iter().find(|t| t.id == id) {
                if !tok.special {
                    bytes.extend_from_slice(tok.content.as_bytes());
                }
                continue;
            }
            let Some(tok) = self.id_to_token.get(&id) else { continue };
            for c in tok.chars() {
                if let Some(&b) = byte_map.get(&c) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        bytes
    }
}

/// GPT-2 ByteLevel pre-tokenizer regex, hand-rolled (leftmost, alternatives in
/// order). Returns the raw substrings; the caller byte-encodes them.
pub fn gpt2_pretokenize(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < chars.len() {
        let len = match_alternative(&chars, pos);
        debug_assert!(len > 0, "no alternative matched at {pos} ({s:?})");
        out.push(chars[pos..pos + len].iter().collect());
        pos += len;
    }
    out
}

fn is_ws(c: char) -> bool {
    c.is_whitespace()
}
fn is_alpha(c: char) -> bool {
    c.is_alphabetic()
}
fn is_num(c: char) -> bool {
    c.is_numeric()
}

/// Length of the first matching regex alternative at `pos` (the GPT-2 pattern).
fn match_alternative(chars: &[char], pos: usize) -> usize {
    let n = chars.len();

    // alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d) — case-insensitive contraction.
    const CONT: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    for c in CONT {
        if pos + c.len() <= n {
            let seg: String = chars[pos..pos + c.len()].iter().collect();
            if seg.eq_ignore_ascii_case(c) {
                return c.len();
            }
        }
    }

    // alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
    {
        let mut i = pos;
        if is_alpha(chars[i]) {
            while i < n && is_alpha(chars[i]) {
                i += 1;
            }
            return i - pos; // >= 1
        }
        // non-alpha: consume one non-(\r\n,L,N) char, then require 1+ alpha.
        if chars[i] != '\r' && chars[i] != '\n' && !is_num(chars[i]) {
            let j = i + 1;
            if j < n && is_alpha(chars[j]) {
                i = j;
                while i < n && is_alpha(chars[i]) {
                    i += 1;
                }
                return i - pos;
            }
        }
    }

    // alt 3: \p{N}
    if is_num(chars[pos]) {
        return 1;
    }

    // alt 4: ?[^\s\p{L}\p{N}]+[\r\n]*  (optional space + punctuation + newlines)
    {
        let mut i = pos;
        if chars[i] == ' ' {
            i += 1;
        }
        let start = i;
        while i < n && !is_ws(chars[i]) && !is_alpha(chars[i]) && !is_num(chars[i]) {
            i += 1;
        }
        if i > start {
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            return i - pos;
        }
    }

    // alt 5: \s*[\r\n]+  — a whitespace run that ends with 1+ newline. `\s*` is
    // greedy then backtracks so `[\r\n]+` lands at the end of the match.
    {
        let mut run_end = pos;
        while run_end < n && is_ws(chars[run_end]) {
            run_end += 1;
        }
        if run_end > pos {
            let mut k = run_end;
            while k > pos && chars[k - 1] != '\r' && chars[k - 1] != '\n' {
                k -= 1;
            }
            if k > pos {
                return k - pos;
            }
        }
    }

    // alt 6: \s+(?!\S)  — a whitespace run not followed by a non-whitespace.
    {
        let mut i = pos;
        while i < n && is_ws(chars[i]) {
            i += 1;
        }
        // greedy `\s+` would be `i`; backtrack to the longest prefix whose next
        // char is NOT non-whitespace (i.e. next is ws or end).
        while i > pos {
            let next = chars.get(i);
            if next.map_or(true, |&c| is_ws(c)) {
                return i - pos;
            }
            i -= 1;
        }
    }

    // alt 7: \s+
    {
        let mut i = pos;
        while i < n && is_ws(chars[i]) {
            i += 1;
        }
        if i > pos {
            return i - pos;
        }
    }

    0
}

/// Map raw substring bytes to the vocab's unicode chars.
fn byte_encode(s: &str) -> String {
    let map = bytes_to_unicode();
    s.bytes().map(|b| map[b as usize]).collect()
}

/// GPT-2 BPE merge: repeatedly merge the adjacent pair with the lowest rank.
fn bpe_merge(word: &str, merges: &HashMap<(String, String), u32>) -> Vec<String> {
    let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    loop {
        let mut best: Option<(usize, u32)> = None;
        for i in 0..tokens.len().saturating_sub(1) {
            let key = (tokens[i].clone(), tokens[i + 1].clone());
            if let Some(&rank) = merges.get(&key) {
                if best.map_or(true, |(_, r)| rank < r) {
                    best = Some((i, rank));
                }
            }
        }
        match best {
            Some((i, _)) => {
                let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                tokens[i] = merged;
                tokens.remove(i + 1);
            }
            None => break,
        }
    }
    tokens
}

/// Load a HF `tokenizer.json` (BPE) into a [`BpeTokenizer`]. Uses the crate's
/// minimal JSON parser; `tokenizer.json` is ~16 MB (151K vocab + 151K merges),
/// so this is a one-time startup cost, not a per-request path.
pub fn from_tokenizer_json(path: &str) -> Result<BpeTokenizer, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let doc = crate::json::parse(&text).map_err(|e| format!("{path}: {e:?}"))?;
    let model = doc_obj(&doc, "model")?;

    let vocab = obj_map(obj_get(model, "vocab")?)?;
    let mut vmap = HashMap::with_capacity(vocab.len());
    for (tok, id) in vocab {
        let id = match id {
            crate::json::JsonValue::Int(n) => *n as u32,
            other => return Err(format!("vocab id not int: {other}")),
        };
        vmap.insert(tok.clone(), id);
    }

    let merges = arr_items(obj_get(model, "merges")?)?;
    let mut mlist = Vec::with_capacity(merges.len());
    for m in merges {
        let pair = arr_items(m)?;
        if pair.len() != 2 {
            return Err("merge not a pair".into());
        }
        let a = pair[0].as_str().ok_or("merge token not str")?.to_string();
        let b = pair[1].as_str().ok_or("merge token not str")?.to_string();
        mlist.push((a, b));
    }

    let added = arr_items(obj_get(&doc, "added_tokens")?)?;
    let mut atoks = Vec::with_capacity(added.len());
    for t in added {
        let id = match obj_get(t, "id")? {
            crate::json::JsonValue::Int(n) => *n as u32,
            other => return Err(format!("added id not int: {other}")),
        };
        let content = obj_get(t, "content")?.as_str().ok_or("added content not str")?.to_string();
        let special = matches!(obj_get(t, "special")?, crate::json::JsonValue::Bool(true));
        atoks.push(AddedToken { content, id, special });
    }

    Ok(BpeTokenizer::from_parts(vmap, mlist, atoks))
}

fn doc_obj<'a>(v: &'a crate::json::JsonValue, key: &str) -> Result<&'a crate::json::JsonValue, String> {
    obj_get(v, key)
}
fn obj_get<'a>(v: &'a crate::json::JsonValue, key: &str) -> Result<&'a crate::json::JsonValue, String> {
    match v {
        crate::json::JsonValue::Object(m) => m.get(key).ok_or_else(|| format!("missing {key}")),
        other => Err(format!("not an object: {other}")),
    }
}
fn obj_map(v: &crate::json::JsonValue) -> Result<&std::collections::BTreeMap<String, crate::json::JsonValue>, String> {
    match v {
        crate::json::JsonValue::Object(m) => Ok(m),
        other => Err(format!("not an object: {other}")),
    }
}
fn arr_items(v: &crate::json::JsonValue) -> Result<&Vec<crate::json::JsonValue>, String> {
    match v {
        crate::json::JsonValue::Array(a) => Ok(a),
        other => Err(format!("not an array: {other}")),
    }
}
