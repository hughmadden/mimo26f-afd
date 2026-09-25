//! zlib (RFC 1950) + DEFLATE (RFC 1951) decoder for PNG image data.
//!
//! Handles stored, fixed-Huffman and dynamic-Huffman blocks and rejects the
//! same malformed streams zlib rejects (over-subscribed / incomplete code
//! sets, missing end-of-block code, bad repeats, distances too far back).
//! Output is bounded by the caller's expected size: decoding stops as soon as
//! `out_len` bytes exist (trailing data is ignored, as Pillow does), so a
//! decompression bomb can never allocate more than the image needs. The
//! Adler-32 trailer is not checked.

use crate::{bail, ImageError};
use std::sync::OnceLock;

/// Inflate a zlib stream, returning exactly `out_len` bytes.
pub(crate) fn zlib_decompress(data: &[u8], out_len: usize) -> Result<Vec<u8>, ImageError> {
    if data.len() < 2 {
        bail!("truncated PNG image data (no zlib header)");
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0F != 8 {
        bail!(
            "PNG image data: unsupported zlib compression method {}",
            cmf & 0x0F
        );
    }
    if cmf >> 4 > 7 {
        bail!("PNG image data: invalid zlib window size");
    }
    if !(((cmf as u16) << 8) | flg as u16).is_multiple_of(31) {
        bail!("PNG image data: bad zlib header check bits");
    }
    if flg & 0x20 != 0 {
        bail!("PNG image data: zlib preset dictionary is not supported");
    }
    inflate(&data[2..], out_len)
}

/// Inflate a raw DEFLATE stream into exactly `out_len` bytes.
pub(crate) fn inflate(data: &[u8], out_len: usize) -> Result<Vec<u8>, ImageError> {
    let mut out = Vec::with_capacity(out_len);
    let mut inf = Inflater {
        data,
        pos: 0,
        bitbuf: 0,
        bitcnt: 0,
        virt: 0,
    };
    if out_len > 0 {
        inf.run(&mut out, out_len)?;
    }
    // Every bit we used must have come from real input, not zero padding.
    let loaded = (inf.pos + inf.virt) as u64 * 8;
    if loaded - inf.bitcnt as u64 > data.len() as u64 * 8 {
        bail!("truncated PNG image data (deflate stream ends early)");
    }
    Ok(out)
}

const TRUNCATED: &str = "truncated PNG image data (deflate stream ends early)";

struct Inflater<'a> {
    data: &'a [u8],
    /// Next input byte to load into the bit buffer.
    pos: usize,
    bitbuf: u64,
    bitcnt: u32,
    /// Zero bytes loaded after the end of `data` (lookahead only).
    virt: usize,
}

impl Inflater<'_> {
    /// Ensure at least 57 bits are buffered. Past the end of the input, zero
    /// bytes are supplied (at most 8, i.e. lookahead only); consuming them is
    /// detected by the final bit-accounting check in `inflate`.
    #[inline(always)]
    fn refill(&mut self) -> Result<(), ImageError> {
        if self.bitcnt > 56 {
            return Ok(());
        }
        if let Some(chunk) = self.data.get(self.pos..self.pos + 8) {
            let word = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            self.bitbuf |= word << self.bitcnt;
            let nbytes = (63 - self.bitcnt) >> 3;
            self.pos += nbytes as usize;
            self.bitcnt += nbytes * 8;
            return Ok(());
        }
        while self.bitcnt <= 56 {
            let byte = match self.data.get(self.pos) {
                Some(&b) => {
                    self.pos += 1;
                    b
                }
                None => {
                    self.virt += 1;
                    if self.virt > 8 {
                        return Err(ImageError::new(TRUNCATED));
                    }
                    0
                }
            };
            self.bitbuf |= (byte as u64) << self.bitcnt;
            self.bitcnt += 8;
        }
        Ok(())
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.bitbuf >>= n;
        self.bitcnt -= n;
    }

    /// Take `n <= 16` bits (caller guarantees they are buffered).
    #[inline(always)]
    fn bits(&mut self, n: u32) -> u32 {
        let v = (self.bitbuf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        v
    }

    fn run(&mut self, out: &mut Vec<u8>, out_len: usize) -> Result<(), ImageError> {
        loop {
            self.refill()?;
            let last = self.bits(1);
            match self.bits(2) {
                0 => self.stored(out, out_len)?,
                1 => {
                    let (lit, dist) = fixed_tables();
                    self.codes(out, out_len, lit, dist)?;
                }
                2 => {
                    let (lit, dist) = self.dynamic_tables()?;
                    self.codes(out, out_len, &lit, &dist)?;
                }
                _ => bail!("PNG image data: invalid deflate block type"),
            }
            if out.len() >= out_len {
                return Ok(());
            }
            if last == 1 {
                bail!(
                    "truncated PNG image data (deflate stream holds {} of {} bytes)",
                    out.len(),
                    out_len
                );
            }
        }
    }

    fn stored(&mut self, out: &mut Vec<u8>, out_len: usize) -> Result<(), ImageError> {
        // Drop to a byte boundary, then hand buffered whole bytes back.
        self.consume(self.bitcnt & 7);
        let mut back = (self.bitcnt / 8) as usize;
        let v = back.min(self.virt);
        self.virt -= v;
        back -= v;
        self.pos -= back;
        self.bitbuf = 0;
        self.bitcnt = 0;
        if self.virt > 0 {
            bail!("{TRUNCATED}");
        }
        let Some(hdr) = self.data.get(self.pos..self.pos + 4) else {
            bail!("{TRUNCATED}");
        };
        let len = u16::from_le_bytes([hdr[0], hdr[1]]);
        let nlen = u16::from_le_bytes([hdr[2], hdr[3]]);
        if len != !nlen {
            bail!("PNG image data: invalid stored block lengths");
        }
        self.pos += 4;
        let want = (len as usize).min(out_len - out.len());
        let avail = self.data.len() - self.pos;
        let take = want.min(avail);
        out.extend_from_slice(&self.data[self.pos..self.pos + take]);
        if out.len() >= out_len {
            self.pos += take;
            return Ok(());
        }
        if avail < len as usize {
            bail!("{TRUNCATED}");
        }
        self.pos += len as usize;
        Ok(())
    }

    fn codes(
        &mut self,
        out: &mut Vec<u8>,
        out_len: usize,
        lit: &Huffman,
        dist: &Huffman,
    ) -> Result<(), ImageError> {
        loop {
            // One refill covers the worst case of 15 + 5 + 15 + 13 = 48 bits.
            self.refill()?;
            let sym = lit.decode(self)?;
            if sym < 256 {
                out.push(sym as u8);
                if out.len() >= out_len {
                    return Ok(());
                }
                continue;
            }
            if sym == 256 {
                return Ok(());
            }
            let li = sym as usize - 257;
            if li >= 29 {
                bail!("PNG image data: invalid literal/length code");
            }
            let len = LEN_BASE[li] as usize + self.bits(LEN_EXTRA[li] as u32) as usize;
            let dsym = dist.decode(self)? as usize;
            if dsym >= 30 {
                bail!("PNG image data: invalid distance code");
            }
            let d = DIST_BASE[dsym] as usize + self.bits(DIST_EXTRA[dsym] as u32) as usize;
            if d > out.len() {
                bail!("PNG image data: invalid distance too far back");
            }
            let len = len.min(out_len - out.len());
            copy_match(out, d, len);
            if out.len() >= out_len {
                return Ok(());
            }
        }
    }

    fn dynamic_tables(&mut self) -> Result<(Huffman, Huffman), ImageError> {
        const ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        self.refill()?;
        let nlen = self.bits(5) as usize + 257;
        let ndist = self.bits(5) as usize + 1;
        let ncode = self.bits(4) as usize + 4;
        if nlen > 286 || ndist > 30 {
            bail!("PNG image data: too many length or distance symbols");
        }
        let mut cl_lens = [0u8; 19];
        for &slot in ORDER.iter().take(ncode) {
            self.refill()?;
            cl_lens[slot] = self.bits(3) as u8;
        }
        let cl = Huffman::new(&cl_lens, Kind::CodeLengths)?;
        let total = nlen + ndist;
        let mut lens = [0u8; 286 + 30];
        let mut i = 0;
        while i < total {
            self.refill()?;
            let sym = cl.decode(self)?;
            let (value, count) = match sym {
                0..=15 => (sym as u8, 1),
                16 => {
                    if i == 0 {
                        bail!("PNG image data: invalid bit length repeat");
                    }
                    (lens[i - 1], 3 + self.bits(2) as usize)
                }
                17 => (0, 3 + self.bits(3) as usize),
                _ => (0, 11 + self.bits(7) as usize),
            };
            if i + count > total {
                bail!("PNG image data: invalid bit length repeat");
            }
            lens[i..i + count].fill(value);
            i += count;
        }
        if lens[256] == 0 {
            bail!("PNG image data: invalid code -- missing end-of-block");
        }
        let lit = Huffman::new(&lens[..nlen], Kind::Lens)?;
        let dist = Huffman::new(&lens[nlen..total], Kind::Dists)?;
        Ok((lit, dist))
    }
}

/// Copy `len` bytes from `dist` back (the regions may overlap).
#[inline(always)]
fn copy_match(out: &mut Vec<u8>, dist: usize, len: usize) {
    let src = out.len() - dist;
    if dist == 1 {
        let b = out[src];
        out.resize(out.len() + len, b);
        return;
    }
    // The output from `src` is periodic with period `dist`; each pass copies
    // everything written since `src`, so passes grow by multiples of `dist`.
    let mut remaining = len;
    while remaining > 0 {
        let k = remaining.min(out.len() - src);
        out.extend_from_within(src..src + k);
        remaining -= k;
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn fixed_tables() -> (&'static Huffman, &'static Huffman) {
    static FIXED: OnceLock<(Huffman, Huffman)> = OnceLock::new();
    let t = FIXED.get_or_init(|| {
        let mut lit = [0u8; 288];
        lit[..144].fill(8);
        lit[144..256].fill(9);
        lit[256..280].fill(7);
        lit[280..].fill(8);
        // zlib's fixed distance table has 32 codes of 5 bits; 30 and 31 decode
        // but are rejected as invalid distance codes.
        let dist = [5u8; 32];
        (
            Huffman::new(&lit, Kind::Lens).expect("fixed literal table"),
            Huffman::new(&dist, Kind::Dists).expect("fixed distance table"),
        )
    });
    (&t.0, &t.1)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    CodeLengths,
    Lens,
    Dists,
}

const FAST_BITS: u32 = 10;

/// Canonical Huffman decoder: a 10-bit direct lookup for short codes and the
/// bit-serial canonical walk (zlib's `puff`) for longer ones.
struct Huffman {
    /// `(symbol << 4) | length` for codes of length <= FAST_BITS, else 0.
    fast: Vec<u16>,
    count: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8], kind: Kind) -> Result<Huffman, ImageError> {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let max = (1..16).rev().find(|&l| count[l] != 0).unwrap_or(0);
        let mut h = Huffman {
            fast: vec![0; 1 << FAST_BITS],
            count,
            symbols: Vec::new(),
        };
        if max == 0 {
            // No symbols: legal (e.g. a distance tree in a literal-only block)
            // but any decode attempt fails.
            return Ok(h);
        }
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left = (left << 1) - c as i32;
            if left < 0 {
                bail!("PNG image data: over-subscribed Huffman code");
            }
        }
        if left > 0 && (kind == Kind::CodeLengths || max != 1) {
            bail!("PNG image data: incomplete Huffman code");
        }
        let mut offs = [0usize; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + count[len] as usize;
        }
        h.symbols = vec![0; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                h.symbols[offs[l as usize]] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        let mut code: u32 = 0;
        let mut k = 0usize;
        for len in 1..=15u32 {
            for _ in 0..count[len as usize] {
                if len <= FAST_BITS {
                    let rev = code.reverse_bits() >> (32 - len);
                    let entry = (h.symbols[k] << 4) | len as u16;
                    let mut i = rev as usize;
                    while i < h.fast.len() {
                        h.fast[i] = entry;
                        i += 1 << len;
                    }
                }
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Ok(h)
    }

    #[inline(always)]
    fn decode(&self, inf: &mut Inflater) -> Result<u16, ImageError> {
        let e = self.fast[(inf.bitbuf & ((1 << FAST_BITS) - 1)) as usize];
        if e != 0 {
            inf.consume((e & 15) as u32);
            return Ok(e >> 4);
        }
        self.decode_slow(inf)
    }

    #[cold]
    fn decode_slow(&self, inf: &mut Inflater) -> Result<u16, ImageError> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= inf.bits(1) as i32;
            let count = self.count[len] as i32;
            if code - count < first {
                if let Some(&s) = self.symbols.get((index + code - first) as usize) {
                    return Ok(s);
                }
                break;
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        bail!("PNG image data: invalid Huffman code")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_block() {
        // zlib.compress(b"hello", 0)
        let z = [
            0x78, 0x01, 0x01, 0x05, 0x00, 0xfa, 0xff, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x06, 0x2c,
            0x02, 0x15,
        ];
        assert_eq!(zlib_decompress(&z, 5).unwrap(), b"hello");
        // Asking for fewer bytes stops early; asking for more is truncation.
        assert_eq!(zlib_decompress(&z, 3).unwrap(), b"hel");
        assert!(zlib_decompress(&z, 6).is_err());
    }

    #[test]
    fn fixed_block_with_overlapping_match() {
        // zlib.compressobj(9, DEFLATED, 15, 9, Z_FIXED) of b"abc" * 8 + b"x" * 40
        let z = [
            0x78, 0x01, 0x4b, 0x4c, 0x4a, 0x4e, 0xc4, 0x86, 0x2a, 0x88, 0x04, 0x00, 0x63, 0x15,
            0x1b, 0xf1,
        ];
        let mut want = b"abcabcabcabcabcabcabcabc".to_vec();
        want.extend(std::iter::repeat_n(b'x', 40));
        assert_eq!(zlib_decompress(&z, want.len()).unwrap(), want);
    }

    #[test]
    fn header_checks() {
        assert!(zlib_decompress(&[0x78], 1).is_err());
        assert!(zlib_decompress(&[0x79, 0x9c, 0, 0], 1).is_err()); // method 9
        assert!(zlib_decompress(&[0x78, 0x9d, 0, 0], 1).is_err()); // check bits
        assert!(zlib_decompress(&[0x78, 0xbb, 0, 0, 0, 0], 1).is_err()); // FDICT
    }

    #[test]
    fn rejects_bad_block_type_and_truncation() {
        // BTYPE=3
        assert!(inflate(&[0x07], 1).is_err());
        // Empty input asking for data.
        assert!(inflate(&[], 1).is_err());
    }
}
