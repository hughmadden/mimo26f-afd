//! Minimal safetensors reader — header parse + tensor byte ranges.
//!
//! Only what the repack tool needs: `U8` tensors, `data_offsets` ranges, and a
//! fail-loud name audit. No mmap, no dtype conversion, no dependency.
//!
//! Format: `<u64 LE header_len><header_len bytes of JSON><tensor data>`, where
//! each header entry is `{"dtype": "U8", "shape": [..], "data_offsets": [a, b]}`
//! with `a`/`b` relative to the start of the data region (i.e. after the 8-byte
//! length and the header).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{io_err, RepackError};

/// One tensor's header entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// safetensors dtype string (must be `"U8"` for expert weights).
    pub dtype: String,
    /// Declared shape.
    pub shape: Vec<usize>,
    /// `[start, end)` byte range relative to the data region.
    pub data_offsets: (u64, u64),
}

impl TensorEntry {
    /// Declared byte length.
    pub fn byte_len(&self) -> u64 {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// A parsed safetensors header plus the absolute offset of its data region.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Tensor name -> entry.
    pub tensors: BTreeMap<String, TensorEntry>,
    /// Absolute file offset where the data region starts.
    pub data_start: u64,
    /// Total file length.
    pub file_len: u64,
}

impl SafetensorsHeader {
    /// Read and parse the header of `path` (READ-ONLY; the file is opened for
    /// reading and never written).
    pub fn read(path: &Path) -> Result<Self, RepackError> {
        let p = path.display().to_string();
        let mut f = File::open(path).map_err(|e| io_err(&p, e))?;
        let file_len = f.metadata().map_err(|e| io_err(&p, e))?.len();
        if file_len < 8 {
            return Err(RepackError::BadSafetensors(format!(
                "{p}: file is {file_len} B, shorter than the 8-byte header length"
            )));
        }
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).map_err(|e| io_err(&p, e))?;
        let header_len = u64::from_le_bytes(len_buf);
        if header_len == 0 || header_len > file_len - 8 {
            return Err(RepackError::BadSafetensors(format!(
                "{p}: header length {header_len} does not fit in {file_len} B"
            )));
        }
        let mut hdr = vec![0u8; header_len as usize];
        f.read_exact(&mut hdr).map_err(|e| io_err(&p, e))?;
        let text = std::str::from_utf8(&hdr)
            .map_err(|e| RepackError::BadSafetensors(format!("{p}: header is not UTF-8: {e}")))?;
        let tensors = parse_header_json(text)
            .map_err(|e| RepackError::BadSafetensors(format!("{p}: {e}")))?;
        let data_start = 8 + header_len;
        // Every declared range must fit inside the file.
        for (name, t) in &tensors {
            let end = data_start + t.data_offsets.1;
            if end > file_len {
                return Err(RepackError::Truncated {
                    path: format!("{p}:{name}"),
                    declared: end,
                    got: file_len,
                });
            }
        }
        Ok(SafetensorsHeader {
            tensors,
            data_start,
            file_len,
        })
    }

    /// Read one tensor's raw bytes.
    pub fn read_tensor(&self, path: &Path, name: &str) -> Result<Vec<u8>, RepackError> {
        let p = path.display().to_string();
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| RepackError::MissingTensor(name.to_string()))?;
        let mut f = File::open(path).map_err(|e| io_err(&p, e))?;
        f.seek(SeekFrom::Start(self.data_start + t.data_offsets.0))
            .map_err(|e| io_err(&p, e))?;
        let mut buf = vec![0u8; t.byte_len() as usize];
        f.read_exact(&mut buf).map_err(|e| io_err(&p, e))?;
        Ok(buf)
    }
}

/// Parse the safetensors header JSON. Deliberately small: it understands the
/// exact shape safetensors emits (`{"name": {"dtype": .., "shape": [..],
/// "data_offsets": [a, b]}, ...}` plus the optional `__metadata__` object) and
/// fails loud on anything else rather than guessing.
pub fn parse_header_json(text: &str) -> Result<BTreeMap<String, TensorEntry>, String> {
    let mut p = JsonParser::new(text);
    p.skip_ws();
    p.expect('{')?;
    let mut out = BTreeMap::new();
    p.skip_ws();
    if p.peek() == Some('}') {
        p.bump();
        // Empty object: the same strict trailing-bytes rule applies here as on
        // the general path below (captain integration fix: the early return
        // used to skip it, so "{} trailing" parsed).
        p.skip_ws();
        if p.pos != text.len() {
            return Err(format!("trailing bytes after header JSON at {}", p.pos));
        }
        return Ok(out);
    }
    loop {
        p.skip_ws();
        let key = p.string()?;
        p.skip_ws();
        p.expect(':')?;
        p.skip_ws();
        if key == "__metadata__" {
            p.skip_value()?;
        } else {
            let entry = p.tensor_entry()?;
            out.insert(key, entry);
        }
        p.skip_ws();
        match p.peek() {
            Some(',') => {
                p.bump();
            }
            Some('}') => {
                p.bump();
                break;
            }
            other => {
                return Err(format!(
                    "expected ',' or '}}' after header entry, got {:?}",
                    other
                ))
            }
        }
    }
    p.skip_ws();
    if p.pos != text.len() {
        return Err(format!("trailing bytes after header JSON at {}", p.pos));
    }
    Ok(out)
}

struct JsonParser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        JsonParser {
            s: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.s.get(self.pos).map(|b| *b as char)
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\n') | Some('\r')) {
            self.bump();
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.bump();
            Ok(())
        } else {
            Err(format!("expected {c:?} at byte {}, got {:?}", self.pos, self.peek()))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| "unterminated string".to_string())?;
            self.bump();
            match c {
                '"' => break,
                '\\' => {
                    let e = self.peek().ok_or_else(|| "unterminated escape".to_string())?;
                    self.bump();
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'u' => {
                            let hex = self
                                .s
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| "short \\u escape".to_string())?;
                            let hex = std::str::from_utf8(hex)
                                .map_err(|_| "bad \\u escape".to_string())?;
                            let cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| format!("bad \\u escape {hex:?}"))?;
                            self.pos += 4;
                            out.push(
                                char::from_u32(cp)
                                    .ok_or_else(|| format!("bad code point {cp:#x}"))?,
                            );
                        }
                        other => return Err(format!("bad escape \\{other}")),
                    }
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }

    fn number(&mut self) -> Result<u64, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.pos {
            return Err(format!("expected a number at byte {}", self.pos));
        }
        std::str::from_utf8(&self.s[start..self.pos])
            .map_err(|_| "bad number".to_string())?
            .parse::<u64>()
            .map_err(|e| format!("bad number: {e}"))
    }

    fn usize_array(&mut self) -> Result<Vec<usize>, String> {
        self.expect('[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(out);
        }
        loop {
            self.skip_ws();
            out.push(self.number()? as usize);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some(']') => {
                    self.bump();
                    break;
                }
                other => return Err(format!("expected ',' or ']', got {other:?}")),
            }
        }
        Ok(out)
    }

    fn tensor_entry(&mut self) -> Result<TensorEntry, String> {
        self.expect('{')?;
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        self.skip_ws();
        if self.peek() == Some('}') {
            self.bump();
            return Err("tensor entry has no fields".to_string());
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(':')?;
            self.skip_ws();
            match key.as_str() {
                "dtype" => dtype = Some(self.string()?),
                "shape" => shape = Some(self.usize_array()?),
                "data_offsets" => {
                    let v = self.usize_array()?;
                    if v.len() != 2 {
                        return Err(format!("data_offsets must have 2 entries, got {}", v.len()));
                    }
                    offsets = Some((v[0] as u64, v[1] as u64));
                }
                _ => self.skip_value()?,
            }
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some('}') => {
                    self.bump();
                    break;
                }
                other => return Err(format!("expected ',' or '}}', got {other:?}")),
            }
        }
        let dtype = dtype.ok_or_else(|| "tensor entry missing dtype".to_string())?;
        let shape = shape.ok_or_else(|| "tensor entry missing shape".to_string())?;
        let data_offsets = offsets.ok_or_else(|| "tensor entry missing data_offsets".to_string())?;
        if data_offsets.1 < data_offsets.0 {
            return Err(format!(
                "data_offsets {:?} are reversed",
                data_offsets
            ));
        }
        Ok(TensorEntry {
            dtype,
            shape,
            data_offsets,
        })
    }

    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => {
                self.string()?;
            }
            Some('{') => {
                self.bump();
                self.skip_ws();
                if self.peek() == Some('}') {
                    self.bump();
                    return Ok(());
                }
                loop {
                    self.skip_ws();
                    self.string()?;
                    self.skip_ws();
                    self.expect(':')?;
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(',') => {
                            self.bump();
                        }
                        Some('}') => {
                            self.bump();
                            break;
                        }
                        other => return Err(format!("expected ',' or '}}', got {other:?}")),
                    }
                }
            }
            Some('[') => {
                self.bump();
                self.skip_ws();
                if self.peek() == Some(']') {
                    self.bump();
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    self.skip_ws();
                    match self.peek() {
                        Some(',') => {
                            self.bump();
                        }
                        Some(']') => {
                            self.bump();
                            break;
                        }
                        other => return Err(format!("expected ',' or ']', got {other:?}")),
                    }
                }
            }
            Some(c) if c == 't' || c == 'f' || c == 'n' => {
                while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic()) {
                    self.bump();
                }
            }
            Some(c) if c.is_ascii_digit() || c == '-' => {
                if c == '-' {
                    self.bump();
                }
                self.number()?;
                if self.peek() == Some('.') {
                    self.bump();
                    self.number()?;
                }
            }
            other => return Err(format!("unexpected {other:?} in JSON value")),
        }
        Ok(())
    }
}
