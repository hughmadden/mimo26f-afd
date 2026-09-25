//! A small strict JSON parser for the manifest reader.
//!
//! Hand-rolled so the crate stays dependency-free. It is deliberately strict:
//! duplicate keys, trailing bytes, non-integer numbers and unescaped control
//! characters are errors. A manifest that cannot be read exactly must fail
//! loud — a lenient parse of an identity manifest is how a wrong slice gets
//! served.

use std::collections::BTreeMap;

use crate::error::RepackError;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// An integer (the manifest has no floats).
    U64(u64),
    /// A string.
    Str(String),
    /// An array.
    Array(Vec<Json>),
    /// An object (keys sorted; duplicate keys are an error).
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// Borrow as an object.
    pub fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow as an array.
    pub fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Borrow as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as an integer.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::U64(n) => Some(*n),
            _ => None,
        }
    }
}

/// Parse a complete JSON document.
pub fn parse(text: &str) -> Result<Json, RepackError> {
    let mut p = P {
        s: text.as_bytes(),
        pos: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.pos != p.s.len() {
        return Err(RepackError::BadManifest(format!(
            "trailing bytes at {}",
            p.pos
        )));
    }
    Ok(v)
}

struct P<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> P<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')) {
            self.bump();
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), RepackError> {
        if self.peek() == Some(c) {
            self.bump();
            Ok(())
        } else {
            Err(RepackError::BadManifest(format!(
                "expected {:?} at byte {}, got {:?}",
                c as char,
                self.pos,
                self.peek().map(|b| b as char)
            )))
        }
    }

    fn lit(&mut self, word: &str) -> Result<(), RepackError> {
        if self.s[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(RepackError::BadManifest(format!(
                "expected {word:?} at byte {}",
                self.pos
            )))
        }
    }

    fn value(&mut self) -> Result<Json, RepackError> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.lit("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.lit("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.lit("null")?;
                Ok(Json::Null)
            }
            Some(c) if c.is_ascii_digit() || c == b'-' => self.number(),
            other => Err(RepackError::BadManifest(format!(
                "unexpected {:?} at byte {}",
                other.map(|b| b as char),
                self.pos
            ))),
        }
    }

    fn object(&mut self) -> Result<Json, RepackError> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(Json::Object(map));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            let v = self.value()?;
            if map.insert(k.clone(), v).is_some() {
                return Err(RepackError::BadManifest(format!("duplicate key {k:?}")));
            }
            self.ws();
            match self.peek() {
                Some(b',') => self.bump(),
                Some(b'}') => {
                    self.bump();
                    break;
                }
                other => {
                    return Err(RepackError::BadManifest(format!(
                        "expected ',' or '}}' at byte {}, got {:?}",
                        self.pos,
                        other.map(|b| b as char)
                    )))
                }
            }
        }
        Ok(Json::Object(map))
    }

    fn array(&mut self) -> Result<Json, RepackError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(Json::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.bump(),
                Some(b']') => {
                    self.bump();
                    break;
                }
                other => {
                    return Err(RepackError::BadManifest(format!(
                        "expected ',' or ']' at byte {}, got {:?}",
                        self.pos,
                        other.map(|b| b as char)
                    )))
                }
            }
        }
        Ok(Json::Array(out))
    }

    fn string(&mut self) -> Result<String, RepackError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| RepackError::BadManifest("unterminated string".into()))?;
            self.bump();
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self
                        .peek()
                        .ok_or_else(|| RepackError::BadManifest("unterminated escape".into()))?;
                    self.bump();
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = self.s.get(self.pos..self.pos + 4).ok_or_else(|| {
                                RepackError::BadManifest("short \\u escape".into())
                            })?;
                            let hex = std::str::from_utf8(hex)
                                .map_err(|_| RepackError::BadManifest("bad \\u escape".into()))?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| {
                                RepackError::BadManifest(format!("bad \\u escape {hex:?}"))
                            })?;
                            self.pos += 4;
                            out.push(char::from_u32(cp).ok_or_else(|| {
                                RepackError::BadManifest(format!("bad code point {cp:#x}"))
                            })?);
                        }
                        other => {
                            return Err(RepackError::BadManifest(format!(
                                "bad escape \\{}",
                                other as char
                            )))
                        }
                    }
                }
                c if c < 0x20 => {
                    return Err(RepackError::BadManifest(format!(
                        "unescaped control byte {c:#04x} in string"
                    )))
                }
                c => out.push(c as char),
            }
        }
        Ok(out)
    }

    fn number(&mut self) -> Result<Json, RepackError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if self.peek() == Some(b'.') || self.peek() == Some(b'e') || self.peek() == Some(b'E') {
            return Err(RepackError::BadManifest(format!(
                "non-integer number at byte {start} (the manifest has no floats)"
            )));
        }
        let text = std::str::from_utf8(&self.s[start..self.pos])
            .map_err(|_| RepackError::BadManifest("bad number bytes".into()))?;
        let n: i64 = text
            .parse()
            .map_err(|e| RepackError::BadManifest(format!("bad number {text:?}: {e}")))?;
        if n < 0 {
            return Err(RepackError::BadManifest(format!(
                "negative number {n} at byte {start}"
            )));
        }
        Ok(Json::U64(n as u64))
    }
}
