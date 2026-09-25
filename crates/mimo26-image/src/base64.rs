//! Base64 (RFC 4648) decoding for data URLs.
//!
//! Standard alphabet; the URL-safe `-`/`_` are also accepted. ASCII
//! whitespace anywhere is skipped. `=` padding is optional but, when present,
//! must be the last non-whitespace characters and consistent with the length.

use crate::{bail, ImageError};

const INVALID: u8 = 0xFF;
const SKIP: u8 = 0xFE;
const PAD: u8 = 0xFD;

const fn build_table() -> [u8; 256] {
    let mut t = [INVALID; 256];
    let mut i = 0;
    while i < 26 {
        t[b'A' as usize + i] = i as u8;
        t[b'a' as usize + i] = 26 + i as u8;
        i += 1;
    }
    let mut d = 0;
    while d < 10 {
        t[b'0' as usize + d] = 52 + d as u8;
        d += 1;
    }
    t[b'+' as usize] = 62;
    t[b'-' as usize] = 62;
    t[b'/' as usize] = 63;
    t[b'_' as usize] = 63;
    t[b'=' as usize] = PAD;
    t[b' ' as usize] = SKIP;
    t[b'\t' as usize] = SKIP;
    t[b'\n' as usize] = SKIP;
    t[b'\r' as usize] = SKIP;
    t[0x0B] = SKIP;
    t[0x0C] = SKIP;
    t
}

static TABLE: [u8; 256] = build_table();

pub(crate) fn decode(input: &[u8]) -> Result<Vec<u8>, ImageError> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3 + 3);
    let mut acc: u32 = 0;
    let mut n = 0u32; // sextets held in `acc`
    let mut pads = 0usize;
    for (i, &c) in input.iter().enumerate() {
        let v = TABLE[c as usize];
        if v < 64 {
            if pads > 0 {
                bail!("invalid base64 in data URL: data after '=' padding at offset {i}");
            }
            acc = (acc << 6) | v as u32;
            n += 1;
            if n == 4 {
                out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]);
                acc = 0;
                n = 0;
            }
        } else if v == SKIP {
            continue;
        } else if v == PAD {
            pads += 1;
        } else {
            bail!("invalid base64 in data URL: byte 0x{c:02x} at offset {i}");
        }
    }
    match (n, pads) {
        (0, 0) => {}
        (2, 0) | (2, 2) => out.push((acc >> 4) as u8),
        (3, 0) | (3, 1) => out.extend_from_slice(&[(acc >> 10) as u8, (acc >> 2) as u8]),
        (1, _) => bail!("invalid base64 in data URL: truncated (length is 1 mod 4)"),
        _ => bail!("invalid base64 in data URL: bad '=' padding"),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn rfc4648_vectors() {
        let v: &[(&str, &str)] = &[
            ("", ""),
            ("Zg==", "f"),
            ("Zm8=", "fo"),
            ("Zm9v", "foo"),
            ("Zm9vYg==", "foob"),
            ("Zm9vYmE=", "fooba"),
            ("Zm9vYmFy", "foobar"),
            ("Zg", "f"),
            ("Zm8", "fo"),
            (" Zm9v\r\nYmFy\n", "foobar"),
        ];
        for (enc, dec) in v {
            assert_eq!(decode(enc.as_bytes()).unwrap(), dec.as_bytes(), "{enc:?}");
        }
    }

    #[test]
    fn all_bytes_roundtrip_url_safe() {
        // "+/" and "-_" decode identically.
        assert_eq!(decode(b"-_8=").unwrap(), decode(b"+/8=").unwrap());
        assert_eq!(decode(b"+/8=").unwrap(), vec![0xfb, 0xff]);
    }

    #[test]
    fn rejects_bad_input() {
        for bad in [
            "Z", "Zm9vY", "Zg=a", "Zg===", "Z===", "Zm9v!", "Zm=9v", "Zm8==",
        ] {
            assert!(decode(bad.as_bytes()).is_err(), "{bad:?} should fail");
        }
    }
}
