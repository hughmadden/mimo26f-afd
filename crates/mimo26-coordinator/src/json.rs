//! Minimal JSON value + recursive-descent parser/serializer, for tool-call
//! argument coercion (T27) and the T24 "JSON-parsable arguments" check. Scoped
//! to what a model's `tojson` renders: null/bool/number/string/array/object.
//! No external dependency (the workspace is deliberately dependency-minimal).

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            JsonValue::Int(n) => Some(*n),
            JsonValue::Float(f) if f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError(pub String);

/// Parse one JSON value, consuming leading whitespace and requiring the whole
/// input (any trailing non-whitespace is an error).
pub fn parse(s: &str) -> Result<JsonValue, JsonError> {
    let mut p = Parser { bytes: s.as_bytes(), pos: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.pos != s.len() {
        return Err(JsonError("trailing bytes after JSON value".into()));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn value(&mut self) -> Result<JsonValue, JsonError> {
        match self.peek() {
            Some(b'n') => self.lit("null", JsonValue::Null),
            Some(b't') => self.lit("true", JsonValue::Bool(true)),
            Some(b'f') => self.lit("false", JsonValue::Bool(false)),
            Some(b'"') => Ok(JsonValue::Str(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            other => Err(JsonError(format!("unexpected byte {:?} at {}", other.map(|b| b as char), self.pos))),
        }
    }

    fn lit(&mut self, word: &str, v: JsonValue) -> Result<JsonValue, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(JsonError(format!("expected {word} at {}", self.pos)))
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        assert_eq!(self.peek(), Some(b'"'));
        self.pos += 1;
        // Accumulate raw UTF-8 bytes: unescaped bytes are copied verbatim
        // (multi-byte UTF-8 stays intact), escapes and \u are re-encoded.
        let mut bytes = Vec::new();
        loop {
            let c = self.bytes.get(self.pos).copied().ok_or_else(|| JsonError("unterminated string".into()))?;
            self.pos += 1;
            match c {
                b'"' => {
                    return String::from_utf8(bytes).map_err(|_| JsonError("invalid utf-8 in string".into()));
                }
                b'\\' => {
                    let e = self.bytes.get(self.pos).copied().ok_or_else(|| JsonError("bad escape".into()))?;
                    self.pos += 1;
                    match e {
                        b'"' => bytes.push(b'"'),
                        b'\\' => bytes.push(b'\\'),
                        b'/' => bytes.push(b'/'),
                        b'b' => bytes.push(0x08),
                        b'f' => bytes.push(0x0c),
                        b'n' => bytes.push(b'\n'),
                        b'r' => bytes.push(b'\r'),
                        b't' => bytes.push(b'\t'),
                        b'u' => {
                            let code = self.hex4()?;
                            // Decode a code unit; combine a high+low surrogate
                            // pair into one scalar.
                            let ch = if (0xD800..=0xDBFF).contains(&code) {
                                // high surrogate: expect \uXXXX low surrogate next
                                if self.bytes.get(self.pos..self.pos + 2) == Some(b"\\u") {
                                    self.pos += 2;
                                    let low = self.hex4()?;
                                    if !(0xDC00..=0xDFFF).contains(&low) {
                                        return Err(JsonError("unpaired high surrogate".into()));
                                    }
                                    let scalar = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                    char::from_u32(scalar).ok_or_else(|| JsonError("bad surrogate pair".into()))?
                                } else {
                                    return Err(JsonError("unpaired high surrogate".into()));
                                }
                            } else {
                                char::from_u32(code).ok_or_else(|| JsonError("bad \\u codepoint".into()))?
                            };
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => return Err(JsonError(format!("bad escape \\{}", other as char))),
                    }
                }
                other => bytes.push(other),
            }
        }
    }

    /// Read 4 hex digits as a u32.
    fn hex4(&mut self) -> Result<u32, JsonError> {
        let hex = self.bytes.get(self.pos..self.pos + 4).ok_or_else(|| JsonError("short \\u escape".into()))?;
        let s = std::str::from_utf8(hex).map_err(|_| JsonError("bad \\u hex".into()))?;
        let code = u32::from_str_radix(s, 16).map_err(|_| JsonError("bad \\u hex".into()))?;
        self.pos += 4;
        Ok(code)
    }

    fn number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| JsonError("bad number".into()))?;
        if is_float {
            text.parse::<f64>().map(JsonValue::Float).map_err(|_| JsonError(format!("bad float {text}")))
        } else {
            text.parse::<i64>().map(JsonValue::Int).map_err(|_| JsonError(format!("bad int {text}")))
        }
    }

    fn array(&mut self) -> Result<JsonValue, JsonError> {
        self.pos += 1; // '['
        self.ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                other => return Err(JsonError(format!("expected , or ] got {:?}", other.map(|b| b as char)))),
            }
        }
    }

    fn object(&mut self) -> Result<JsonValue, JsonError> {
        self.pos += 1; // '{'
        self.ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(JsonError("object key must be a string".into()));
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err(JsonError("expected : after object key".into()));
            }
            self.pos += 1;
            self.ws();
            let v = self.value()?;
            map.insert(key, v);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                other => return Err(JsonError(format!("expected , or }} got {:?}", other.map(|b| b as char)))),
            }
        }
    }
}

/// Serialize a value to compact JSON.
pub fn serialize(v: &JsonValue) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn write_json(v: &JsonValue, out: &mut String) {
    match v {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Int(n) => out.push_str(&n.to_string()),
        JsonValue::Float(f) => out.push_str(&fmt_f(*f)),
        JsonValue::Str(s) => write_string(s, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        JsonValue::Object(map) => {
            out.push('{');
            for (i, (k, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_json(val, out);
            }
            out.push('}');
        }
    }
}

fn fmt_f(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 {
        format!("{:.1}", f)
    } else {
        format!("{}", f)
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&serialize(self))
    }
}
