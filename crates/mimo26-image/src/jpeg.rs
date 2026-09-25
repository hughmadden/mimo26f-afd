//! Baseline/extended-sequential and progressive Huffman JPEG decoder that
//! reproduces libjpeg-turbo 3.1 as Pillow 12 drives it (`draft` off):
//!
//! * entropy decoding as `jdhuff.c` / `jdphuff.c` (natural-order overflow
//!   slots, restart resync, "hit a marker" zero-fill of the rest of the
//!   restart interval, standard K.3 tables when a sequential file has no DHT);
//! * the ISLOW IDCT as Pillow runs it on x86-64 (`jsimd_idct_islow_avx2`):
//!   identical to the C `jpeg_idct_islow` for in-range coefficients, and
//!   reproducing its 16-bit wrap / saturation for extreme ones (see IDCT);
//! * `jdsample.c` upsampling with `do_fancy_upsampling`: triangular h2v1
//!   (width > 2), h1v2, h2v2 (width > 2), box replication otherwise;
//! * `jdcolor.c` JFIF YCbCr->RGB integer tables; colorspace chosen like
//!   `default_decompress_parms` (JFIF > Adobe transform > component ids);
//! * `jdcoefct.c` inter-block smoothing (`do_block_smoothing`, on by default)
//!   for progressive images whose first ten coefficients are not fully
//!   refined (unusual scan scripts, or a truncated file closed by an EOI),
//!   including its `last_good_iMCU_row` / previous-scan `coef_bits` switch.
//!
//! Not supported (clear errors): arithmetic coding, lossless and hierarchical
//! frames, precision other than 8 bits, 4-component (CMYK/YCCK) and 2-component
//! images. Corrupt entropy data (bad Huffman codes, truncation without an EOI)
//! is an error; libjpeg would warn and continue on some of these.

use crate::exif;
use crate::{bail, ImageError, RgbImage, MAX_DECODE_PIXELS};
use std::sync::OnceLock;

pub(crate) fn decode(data: &[u8]) -> Result<RgbImage, ImageError> {
    let mut p = Parser::new(data);
    let img = p.run()?;
    let o = exif::orientation(p.exif.as_deref(), p.xmp.as_deref());
    Ok(exif::apply(img, o))
}

/// `jpeg_natural_order` plus 16 overflow entries (corrupt run lengths land on 63).
const NATURAL: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Color {
    Gray,
    YCbCr,
    Rgb,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Up {
    Full,
    H2V1Fancy,
    H1V2Fancy,
    H2V2Fancy,
    /// Box replication by (h, v) factors (`h2v1_upsample`, `h2v2_upsample`,
    /// `int_upsample`).
    Box(usize, usize),
}

struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Samples of this component inside the image (`downsampled_width/height`).
    dw: usize,
    dh: usize,
    /// Blocks covering the image area (non-interleaved scan extent).
    bw: usize,
    bh: usize,
    /// MCU-padded block grid (interleaved scan extent / buffer size).
    pbw: usize,
    pbh: usize,
    up: Up,
    /// Quantization table latched at the component's first scan, natural
    /// order, as libjpeg's 16-bit `ISLOW_MULT_TYPE`.
    q: Option<[i32; 64]>,
    pred: i32,
    /// Progressive coefficient buffer (`pbw * pbh * 64`).
    coefs: Vec<i16>,
    /// Sample plane, stride `pbw * 8`, `pbh * 8` rows.
    plane: Vec<u8>,
    /// The same table as stored (`quantval`, u16), for block smoothing.
    qraw: [u16; 64],
    /// Progressive `coef_bits` (-1 = never coded) and the values before the
    /// component's latest scan (`coef_bits[ci + num_components]`).
    coef_bits: [i32; 64],
    prev_coef_bits: [i32; 64],
    scanned: bool,
}

impl Comp {
    fn stride(&self) -> usize {
        self.pbw * 8
    }
}

struct Frame {
    width: usize,
    height: usize,
    progressive: bool,
    comps: Vec<Comp>,
    mcus_x: usize,
    mcus_y: usize,
    /// `master->last_good_iMCU_row`: iMCU row of the last MCU fetched while
    /// entropy data was still sufficient.
    last_good: usize,
}

#[derive(Clone)]
struct RawHuff {
    bits: [u8; 17],
    vals: [u8; 256],
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    qt: [Option<[u16; 64]>; 4],
    /// `[0]` = DC tables, `[1]` = AC tables.
    dht: [[Option<RawHuff>; 4]; 2],
    restart_interval: usize,
    frame: Option<Frame>,
    jfif: bool,
    adobe: Option<u8>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    scans: usize,
    color: Color,
    /// Sequential frame whose first scan held every component (libjpeg is
    /// then in single-scan mode and rejects further scans).
    single_scan: bool,
}

impl<'a> Parser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Parser {
            data,
            pos: 2,
            qt: [None; 4],
            dht: Default::default(),
            restart_interval: 0,
            frame: None,
            jfif: false,
            adobe: None,
            exif: None,
            xmp: None,
            scans: 0,
            color: Color::YCbCr,
            single_scan: false,
        }
    }

    /// libjpeg `next_marker`: skip garbage and 0xFF fill; FF00 is garbage here.
    fn next_marker(&mut self) -> Result<u8, ImageError> {
        let d = self.data;
        loop {
            while self.pos < d.len() && d[self.pos] != 0xFF {
                self.pos += 1;
            }
            while self.pos < d.len() && d[self.pos] == 0xFF {
                self.pos += 1;
            }
            let Some(&m) = d.get(self.pos) else {
                if self.scans == 0 {
                    bail!("truncated JPEG (no image data)");
                }
                bail!("truncated JPEG (missing EOI marker)");
            };
            self.pos += 1;
            if m != 0 {
                return Ok(m);
            }
        }
    }

    fn segment(&mut self) -> Result<&'a [u8], ImageError> {
        let d = self.data;
        let Some(l) = d.get(self.pos..self.pos + 2) else {
            bail!("truncated JPEG (marker segment)");
        };
        let len = u16::from_be_bytes([l[0], l[1]]) as usize;
        if len < 2 {
            bail!("corrupt JPEG: bad marker segment length");
        }
        let Some(seg) = d.get(self.pos + 2..self.pos + len) else {
            bail!("truncated JPEG (marker segment)");
        };
        self.pos += len;
        Ok(seg)
    }

    fn run(&mut self) -> Result<RgbImage, ImageError> {
        loop {
            let m = self.next_marker()?;
            match m {
                0xD9 => break,
                0xD8 => bail!("corrupt JPEG: duplicate SOI marker"),
                0xC0..=0xC2 => {
                    let s = self.segment()?;
                    self.sof(m, s)?;
                }
                0xC3 => bail!("lossless JPEG is not supported (baseline and progressive only)"),
                0xC5..=0xC7 => {
                    bail!("hierarchical JPEG is not supported (baseline and progressive only)")
                }
                0xC9..=0xCB | 0xCD..=0xCF => {
                    bail!("arithmetic-coded JPEG is not supported (Huffman only)")
                }
                0xC4 => {
                    let s = self.segment()?;
                    self.dht(s)?;
                }
                0xDB => {
                    let s = self.segment()?;
                    self.dqt(s)?;
                }
                0xDD => {
                    let s = self.segment()?;
                    if s.len() != 2 {
                        bail!("corrupt JPEG: bad DRI length");
                    }
                    self.restart_interval = u16::from_be_bytes([s[0], s[1]]) as usize;
                }
                0xDA => {
                    let s = self.segment()?;
                    self.sos(s)?;
                }
                0xE0..=0xEF => {
                    let s = self.segment()?;
                    self.app(m, s);
                }
                // Parameterless markers outside a scan are ignored.
                0xD0..=0xD7 | 0x01 => {}
                // COM, DNL, DAC: skipped.
                0xFE | 0xDC | 0xCC => {
                    self.segment()?;
                }
                _ => bail!("corrupt JPEG: unsupported marker 0xFF{m:02X}"),
            }
        }
        if self.scans == 0 {
            bail!("JPEG has no image data");
        }
        let Some(frame) = self.frame.as_mut() else {
            bail!("JPEG has no frame header");
        };
        finish(frame, self.color, self.scans)
    }

    fn app(&mut self, m: u8, s: &[u8]) {
        // Pillow parses markers only up to the first SOS; libjpeg fixes the
        // colorspace at the first SOS too.
        if self.scans > 0 {
            return;
        }
        match m {
            0xE0 if s.len() >= 14 && s.starts_with(b"JFIF\0") => self.jfif = true,
            0xEE if s.len() >= 12 && s.starts_with(b"Adobe") => self.adobe = Some(s[11]),
            0xE1 if s.starts_with(b"Exif\0\0") => match &mut self.exif {
                // Pillow 12 concatenates the payloads of later Exif segments.
                Some(e) => e.extend_from_slice(&s[6..]),
                None => self.exif = Some(s.to_vec()),
            },
            0xE1 if s.starts_with(b"http://ns.adobe.com/xap/1.0/\0") => {
                let nul = s.iter().position(|&b| b == 0).unwrap_or(s.len() - 1);
                self.xmp = Some(s[nul + 1..].to_vec());
            }
            _ => {}
        }
    }

    fn dqt(&mut self, mut s: &[u8]) -> Result<(), ImageError> {
        while !s.is_empty() {
            let (pq, tq) = ((s[0] >> 4) as usize, (s[0] & 15) as usize);
            if tq >= 4 {
                bail!("corrupt JPEG: bad quantization table index {tq}");
            }
            let n = if pq != 0 { 128 } else { 64 };
            let Some(body) = s.get(1..1 + n) else {
                bail!("corrupt JPEG: truncated quantization table");
            };
            let mut t = [0u16; 64];
            for (i, &nat) in NATURAL[..64].iter().enumerate() {
                t[nat] = if pq != 0 {
                    u16::from_be_bytes([body[2 * i], body[2 * i + 1]])
                } else {
                    body[i] as u16
                };
            }
            self.qt[tq] = Some(t);
            s = &s[1 + n..];
        }
        Ok(())
    }

    fn dht(&mut self, mut s: &[u8]) -> Result<(), ImageError> {
        while !s.is_empty() {
            if s.len() < 17 {
                bail!("corrupt JPEG: truncated Huffman table");
            }
            let index = s[0];
            let mut bits = [0u8; 17];
            bits[1..].copy_from_slice(&s[1..17]);
            let count: usize = bits.iter().map(|&b| b as usize).sum();
            if count > 256 || count > s.len() - 17 {
                bail!("corrupt JPEG: bad Huffman table definition");
            }
            let mut vals = [0u8; 256];
            vals[..count].copy_from_slice(&s[17..17 + count]);
            let (class, slot) = if index & 0x10 != 0 {
                (1, index - 0x10)
            } else {
                (0, index)
            };
            if slot >= 4 {
                bail!("corrupt JPEG: bad Huffman table index {index}");
            }
            self.dht[class][slot as usize] = Some(RawHuff { bits, vals });
            s = &s[17 + count..];
        }
        Ok(())
    }

    fn sof(&mut self, m: u8, s: &[u8]) -> Result<(), ImageError> {
        if self.frame.is_some() {
            bail!("corrupt JPEG: more than one frame header");
        }
        if s.len() < 6 {
            bail!("corrupt JPEG: short frame header");
        }
        let precision = s[0];
        let height = u16::from_be_bytes([s[1], s[2]]) as usize;
        let width = u16::from_be_bytes([s[3], s[4]]) as usize;
        let nc = s[5] as usize;
        if precision != 8 {
            bail!("{precision}-bit JPEG is not supported (8-bit only)");
        }
        match nc {
            1 | 3 => {}
            4 => bail!("CMYK/YCCK JPEG (4 components) is not supported"),
            n => bail!("JPEG with {n} components is not supported"),
        }
        if s.len() != 6 + 3 * nc {
            bail!("corrupt JPEG: bad frame header length");
        }
        if width == 0 || height == 0 {
            bail!("JPEG has zero width or height ({width}x{height}; DNL is not supported)");
        }
        if width > 65500 || height > 65500 {
            // libjpeg JPEG_MAX_DIMENSION (JERR_IMAGE_TOO_BIG)
            bail!("JPEG is too large: {width}x{height} exceeds the 65500 pixel side limit");
        }
        if (width as u64) * (height as u64) > MAX_DECODE_PIXELS {
            bail!(
                "JPEG is too large: {width}x{height} exceeds the {MAX_DECODE_PIXELS} pixel limit"
            );
        }
        let mut comps: Vec<Comp> = Vec::with_capacity(nc);
        for i in 0..nc {
            let c = &s[6 + 3 * i..9 + 3 * i];
            let (h, v) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                bail!("corrupt JPEG: bad sampling factors {h}x{v}");
            }
            if c[2] >= 4 {
                bail!("corrupt JPEG: bad quantization table index {}", c[2]);
            }
            if comps.iter().any(|o| o.id == c[0]) {
                bail!("corrupt JPEG: duplicate component id {}", c[0]);
            }
            comps.push(Comp {
                id: c[0],
                h,
                v,
                tq: c[2] as usize,
                dw: 0,
                dh: 0,
                bw: 0,
                bh: 0,
                pbw: 0,
                pbh: 0,
                up: Up::Full,
                q: None,
                pred: 0,
                coefs: Vec::new(),
                plane: Vec::new(),
                qraw: [0; 64],
                coef_bits: [-1; 64],
                prev_coef_bits: [0; 64],
                scanned: false,
            });
        }
        let max_h = comps.iter().map(|c| c.h).max().unwrap_or(1);
        let max_v = comps.iter().map(|c| c.v).max().unwrap_or(1);
        let mcus_x = width.div_ceil(8 * max_h);
        let mcus_y = height.div_ceil(8 * max_v);
        for c in &mut comps {
            if max_h % c.h != 0 || max_v % c.v != 0 {
                bail!("unsupported JPEG sampling factors (non-integral upsampling ratio)");
            }
            c.dw = (width * c.h).div_ceil(max_h);
            c.dh = (height * c.v).div_ceil(max_v);
            c.bw = (width * c.h).div_ceil(max_h * 8);
            c.bh = (height * c.v).div_ceil(max_v * 8);
            c.pbw = mcus_x * c.h;
            c.pbh = mcus_y * c.v;
            let (he, ve) = (max_h / c.h, max_v / c.v);
            c.up = match (he, ve) {
                (1, 1) => Up::Full,
                (2, 1) if c.dw > 2 => Up::H2V1Fancy,
                (1, 2) => Up::H1V2Fancy,
                (2, 2) if c.dw > 2 => Up::H2V2Fancy,
                (he, ve) => Up::Box(he, ve),
            };
        }
        self.frame = Some(Frame {
            width,
            height,
            progressive: m == 0xC2,
            comps,
            mcus_x,
            mcus_y,
            last_good: 0,
        });
        Ok(())
    }

    fn sos(&mut self, s: &[u8]) -> Result<(), ImageError> {
        let first_scan = self.scans == 0;
        let Some(frame) = self.frame.as_mut() else {
            bail!("corrupt JPEG: scan before frame header");
        };
        let ns = *s.first().unwrap_or(&0) as usize;
        if !(1..=4).contains(&ns) || s.len() != 4 + 2 * ns {
            bail!("corrupt JPEG: bad scan header");
        }
        let mut sc: Vec<(usize, usize, usize)> = Vec::with_capacity(ns);
        for i in 0..ns {
            let (id, t) = (s[1 + 2 * i], s[2 + 2 * i]);
            let Some(ci) = frame.comps.iter().position(|c| c.id == id) else {
                bail!("corrupt JPEG: scan references unknown component {id}");
            };
            if sc.iter().any(|&(c, _, _)| c == ci) {
                bail!("corrupt JPEG: component {id} repeated in scan");
            }
            sc.push((ci, (t >> 4) as usize, (t & 15) as usize));
        }
        let (ss, se) = (s[1 + 2 * ns] as usize, s[2 + 2 * ns] as usize);
        let (ah, al) = ((s[3 + 2 * ns] >> 4) as u32, (s[3 + 2 * ns] & 15) as u32);

        if first_scan {
            // default_decompress_parms, then std_huff_tables for empty slots.
            self.color = match frame.comps.len() {
                1 => Color::Gray,
                _ => {
                    let ids: Vec<u8> = frame.comps.iter().map(|c| c.id).collect();
                    if self.jfif {
                        Color::YCbCr
                    } else if let Some(t) = self.adobe {
                        if t == 0 {
                            Color::Rgb
                        } else {
                            Color::YCbCr
                        }
                    } else if ids == [82, 71, 66] {
                        Color::Rgb
                    } else {
                        Color::YCbCr
                    }
                }
            };
            // jinit_huff_decoder (sequential only; jdphuff has no defaults).
            for (class, slot, bits, vals) in std_tables() {
                if !frame.progressive && self.dht[class][slot].is_none() {
                    let mut b = [0u8; 17];
                    b[1..].copy_from_slice(bits);
                    let mut v = [0u8; 256];
                    v[..vals.len()].copy_from_slice(vals);
                    self.dht[class][slot] = Some(RawHuff { bits: b, vals: v });
                }
            }
            if !frame.progressive && ns == frame.comps.len() {
                self.single_scan = true;
            }
            // Planes (and coefficient buffers) are allocated now, not at SOF.
            for c in &mut frame.comps {
                let (bw, bh) = (c.pbw, c.pbh);
                c.plane = vec![0u8; bw * 8 * bh * 8];
                if frame.progressive {
                    c.coefs = vec![0i16; bw * bh * 64];
                }
            }
        } else if self.single_scan {
            bail!("corrupt JPEG: unexpected extra scan in a single-scan image");
        }

        // Latch quantization tables (libjpeg `latch_quant_tables`).
        for &(ci, _, _) in &sc {
            let c = &mut frame.comps[ci];
            if c.q.is_none() {
                let Some(t) = self.qt[c.tq] else {
                    bail!("corrupt JPEG: quantization table {} is missing", c.tq);
                };
                let mut q = [0i32; 64];
                for (d, &s) in q.iter_mut().zip(t.iter()) {
                    *d = s as i16 as i32;
                }
                c.q = Some(q);
                c.qraw = t;
            }
            c.scanned = true;
        }

        let mut blocks = 0;
        for &(ci, _, _) in &sc {
            blocks += frame.comps[ci].h * frame.comps[ci].v;
        }
        if ns > 1 && blocks > 10 {
            bail!("corrupt JPEG: too many blocks in an MCU ({blocks})");
        }

        let kind = if frame.progressive {
            let dc_band = ss == 0;
            let bad = if dc_band {
                se != 0
            } else {
                ss > se || se >= 64 || ns != 1
            } || (ah != 0 && al + 1 != ah)
                || al > 13;
            if bad {
                bail!("corrupt JPEG: invalid progressive scan (Ss={ss} Se={se} Ah={ah} Al={al})");
            }
            // jdphuff start_pass: remember the previous state, then record Al.
            let scan_number = self.scans + 1;
            for &(ci, _, _) in &sc {
                let c = &mut frame.comps[ci];
                for k in ss.min(1)..=se.max(9) {
                    c.prev_coef_bits[k] = if scan_number > 1 { c.coef_bits[k] } else { 0 };
                }
                for k in ss..=se {
                    c.coef_bits[k] = al as i32;
                }
            }
            match (dc_band, ah == 0) {
                (true, true) => ScanKind::DcFirst,
                (true, false) => ScanKind::DcRefine,
                (false, true) => ScanKind::AcFirst,
                (false, false) => ScanKind::AcRefine,
            }
        } else {
            ScanKind::Sequential
        };

        // Derived Huffman tables (`jpeg_make_d_derived_tbl`) for this scan.
        let mut dc_t: Vec<Option<Huff>> = Vec::with_capacity(ns);
        let mut ac_t: Vec<Option<Huff>> = Vec::with_capacity(ns);
        for &(_, td, ta) in &sc {
            let need_dc = matches!(kind, ScanKind::Sequential | ScanKind::DcFirst);
            let need_ac = matches!(
                kind,
                ScanKind::Sequential | ScanKind::AcFirst | ScanKind::AcRefine
            );
            dc_t.push(if need_dc {
                Some(derived(&self.dht, 0, td)?)
            } else {
                None
            });
            ac_t.push(if need_ac {
                Some(derived(&self.dht, 1, ta)?)
            } else {
                None
            });
        }

        let scan = Scan {
            comps: sc.iter().map(|&(ci, _, _)| ci).collect(),
            kind,
            ss,
            se,
            al,
        };
        let pos = decode_scan(
            self.data,
            self.pos,
            self.restart_interval,
            frame,
            &scan,
            &dc_t,
            &ac_t,
        )?;
        self.pos = pos;
        self.scans += 1;
        Ok(())
    }
}

fn derived(dht: &[[Option<RawHuff>; 4]; 2], class: usize, slot: usize) -> Result<Huff, ImageError> {
    let Some(raw) = dht[class].get(slot).and_then(|t| t.as_ref()) else {
        bail!("corrupt JPEG: Huffman table {slot} is missing");
    };
    Huff::new(raw, class == 0)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanKind {
    Sequential,
    DcFirst,
    DcRefine,
    AcFirst,
    AcRefine,
}

struct Scan {
    comps: Vec<usize>,
    kind: ScanKind,
    ss: usize,
    se: usize,
    al: u32,
}

const LOOKAHEAD: u32 = 9;

/// `d_derived_tbl`: 9-bit lookahead plus maxcode/valoffset for longer codes.
struct Huff {
    /// `(length << 8) | symbol`, 0 when the code is longer than LOOKAHEAD.
    lookup: [u16; 1 << LOOKAHEAD],
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    vals: [u8; 256],
}

impl Huff {
    fn new(raw: &RawHuff, is_dc: bool) -> Result<Huff, ImageError> {
        let mut huffsize: Vec<u32> = Vec::with_capacity(257);
        for l in 1..=16u32 {
            for _ in 0..raw.bits[l as usize] {
                huffsize.push(l);
            }
        }
        if huffsize.len() > 256 {
            bail!("corrupt JPEG: bad Huffman table");
        }
        let n = huffsize.len();
        let mut huffcode = vec![0u32; n];
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        let mut p = 0;
        while p < n {
            while p < n && huffsize[p] == si {
                huffcode[p] = code;
                p += 1;
                code += 1;
            }
            if code as u64 >= 1u64 << si {
                bail!("corrupt JPEG: bad Huffman table");
            }
            code <<= 1;
            si += 1;
        }
        let mut t = Huff {
            lookup: [0; 1 << LOOKAHEAD],
            maxcode: [-1; 18],
            valoffset: [0; 18],
            vals: raw.vals,
        };
        let mut p = 0usize;
        for l in 1..=16usize {
            let b = raw.bits[l] as usize;
            if b > 0 {
                t.valoffset[l] = p as i32 - huffcode[p] as i32;
                p += b;
                t.maxcode[l] = huffcode[p - 1] as i32;
            }
        }
        t.maxcode[17] = 0xFFFFF;
        for (i, (&size, &c)) in huffsize.iter().zip(huffcode.iter()).enumerate() {
            if size <= LOOKAHEAD {
                let shift = LOOKAHEAD - size;
                let base = (c << shift) as usize;
                for k in 0..(1usize << shift) {
                    t.lookup[base + k] = ((size as u16) << 8) | raw.vals[i] as u16;
                }
            }
        }
        if is_dc && raw.vals[..n].iter().any(|&v| v > 15) {
            bail!("corrupt JPEG: bad DC Huffman table");
        }
        Ok(t)
    }
}

/// Entropy-coded segment reader with libjpeg's marker/zero-fill behaviour.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    /// MSB-aligned bit accumulator; the top `nbits` bits are valid.
    acc: u64,
    nbits: u32,
    /// Zero bits appended after a marker / end of data.
    fake: u32,
    /// Position of the first 0xFF of the marker that ended the data.
    marker_at: Option<usize>,
    eof: bool,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Bits {
            data,
            pos,
            acc: 0,
            nbits: 0,
            fake: 0,
            marker_at: None,
            eof: false,
        }
    }

    #[inline(always)]
    fn fill(&mut self) {
        while self.nbits <= 56 {
            if self.marker_at.is_some() || self.eof {
                self.nbits += 8;
                self.fake += 8;
                continue;
            }
            let Some(&b) = self.data.get(self.pos) else {
                self.eof = true;
                continue;
            };
            if b != 0xFF {
                self.pos += 1;
                self.acc |= (b as u64) << (56 - self.nbits);
                self.nbits += 8;
                continue;
            }
            let mut p = self.pos + 1;
            while p < self.data.len() && self.data[p] == 0xFF {
                p += 1;
            }
            match self.data.get(p) {
                None => self.eof = true,
                Some(0) => {
                    self.pos = p + 1;
                    self.acc |= 0xFFu64 << (56 - self.nbits);
                    self.nbits += 8;
                }
                Some(_) => self.marker_at = Some(self.pos),
            }
        }
    }

    #[inline(always)]
    fn ensure(&mut self, n: u32) {
        if self.nbits < n {
            self.fill();
        }
    }

    #[inline(always)]
    fn peek(&self, n: u32) -> u32 {
        (self.acc >> (64 - n)) as u32
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.acc <<= n;
        self.nbits -= n;
    }

    #[inline(always)]
    fn get(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        self.ensure(n);
        let v = self.peek(n);
        self.consume(n);
        v
    }

    /// Bits consumed beyond the real data (zero fill was used).
    fn overrun(&self) -> bool {
        self.fake > self.nbits
    }

    #[inline(always)]
    fn huff(&mut self, t: &Huff) -> Result<u8, ImageError> {
        self.ensure(16);
        let e = t.lookup[self.peek(LOOKAHEAD) as usize];
        if e != 0 {
            self.consume((e >> 8) as u32);
            return Ok(e as u8);
        }
        let mut l = LOOKAHEAD + 1;
        while l <= 16 {
            let code = self.peek(l) as i32;
            if code <= t.maxcode[l as usize] {
                self.consume(l);
                return Ok(t.vals[((code + t.valoffset[l as usize]) & 0xFF) as usize]);
            }
            l += 1;
        }
        bail!("corrupt JPEG data: bad Huffman code")
    }

    fn reset_bits(&mut self) {
        self.acc = 0;
        self.nbits = 0;
        self.fake = 0;
    }

    /// Next marker at or after `pos` (skipping garbage): (FF position, code).
    fn find_marker(&self, mut p: usize) -> Option<(usize, u8)> {
        let d = self.data;
        loop {
            while p < d.len() && d[p] != 0xFF {
                p += 1;
            }
            let start = p;
            while p < d.len() && d[p] == 0xFF {
                p += 1;
            }
            let &m = d.get(p)?;
            if m != 0 {
                return Some((start, m));
            }
            p += 1;
        }
    }

    fn marker_code(&self, at: usize) -> Option<(usize, u8)> {
        self.find_marker(at)
    }

    /// `process_restart` + `read_restart_marker` + `jpeg_resync_to_restart`.
    /// Returns true when the marker was left unread (resync action 3).
    fn restart(&mut self, expected: u8) -> Result<bool, ImageError> {
        self.reset_bits();
        let found = match self.marker_at {
            Some(at) => self.marker_code(at),
            None => self.find_marker(self.pos),
        };
        let Some((mut at, mut code)) = found else {
            bail!("truncated JPEG (missing restart marker)");
        };
        loop {
            let after = self.skip_marker(at);
            if code == 0xD0 + expected {
                self.consume_marker(after);
                return Ok(false);
            }
            let action = if code < 0xC0 {
                2
            } else if !(0xD0..=0xD7).contains(&code) {
                3
            } else {
                let n = code - 0xD0;
                if n == (expected + 1) & 7 || n == (expected + 2) & 7 {
                    3
                } else if n == (expected + 7) & 7 || n == (expected + 6) & 7 {
                    2
                } else {
                    1
                }
            };
            match action {
                1 => {
                    self.consume_marker(after);
                    return Ok(false);
                }
                2 => match self.find_marker(after) {
                    Some((a, c)) => {
                        at = a;
                        code = c;
                    }
                    None => bail!("truncated JPEG (missing restart marker)"),
                },
                _ => {
                    self.pos = at;
                    self.marker_at = Some(at);
                    self.eof = false;
                    return Ok(true);
                }
            }
        }
    }

    /// Position just after the marker code whose FF sequence starts at `at`.
    fn skip_marker(&self, at: usize) -> usize {
        let mut p = at;
        while p < self.data.len() && self.data[p] == 0xFF {
            p += 1;
        }
        p + 1
    }

    fn consume_marker(&mut self, after: usize) {
        self.pos = after;
        self.marker_at = None;
        self.eof = false;
    }
}

#[inline(always)]
fn extend(v: u32, s: u32) -> i32 {
    // HUFF_EXTEND
    let v = v as i32;
    if s == 0 {
        0
    } else if v < (1 << (s - 1)) {
        v - (1 << s) + 1
    } else {
        v
    }
}

/// Decode one scan starting at `pos`; returns where marker parsing resumes.
fn decode_scan(
    data: &[u8],
    pos: usize,
    restart_interval: usize,
    frame: &mut Frame,
    scan: &Scan,
    dc_t: &[Option<Huff>],
    ac_t: &[Option<Huff>],
) -> Result<usize, ImageError> {
    let mut br = Bits::new(data, pos);
    let ns = scan.comps.len();
    let (mcus_x, mcus_y) = if ns == 1 {
        let c = &frame.comps[scan.comps[0]];
        (c.bw, c.bh)
    } else {
        (frame.mcus_x, frame.mcus_y)
    };
    for &ci in &scan.comps {
        frame.comps[ci].pred = 0;
    }
    let mut eobrun: u32 = 0;
    let mut insufficient = false;
    let mut restarts_left = restart_interval;
    let mut next_rst: u8 = 0;

    let rows_per_imcu = if ns == 1 {
        frame.comps[scan.comps[0]].v
    } else {
        1
    };
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // jdcoefct consume_data: checked before decode_mcu (and so before
            // its restart processing).
            if !insufficient {
                frame.last_good = my / rows_per_imcu;
            }
            if restart_interval > 0 {
                if restarts_left == 0 {
                    let pending = br.restart(next_rst)?;
                    next_rst = (next_rst + 1) & 7;
                    if !pending {
                        insufficient = false;
                    }
                    for &ci in &scan.comps {
                        frame.comps[ci].pred = 0;
                    }
                    eobrun = 0;
                    restarts_left = restart_interval;
                }
                restarts_left -= 1;
            }
            for (si, &ci) in scan.comps.iter().enumerate() {
                let c = &mut frame.comps[ci];
                let (nv, nh) = if ns == 1 { (1, 1) } else { (c.v, c.h) };
                for by in 0..nv {
                    for bx in 0..nh {
                        let (row, col) = if ns == 1 {
                            (my, mx)
                        } else {
                            (my * c.v + by, mx * c.h + bx)
                        };
                        match scan.kind {
                            ScanKind::Sequential => {
                                let mut blk = [0i16; 64];
                                if !insufficient {
                                    let (Some(dc), Some(ac)) = (&dc_t[si], &ac_t[si]) else {
                                        bail!("corrupt JPEG: missing Huffman table");
                                    };
                                    seq_block(&mut br, dc, ac, &mut c.pred, &mut blk)?;
                                }
                                let stride = c.stride();
                                let q = c.q.as_ref().expect("latched");
                                let off = row * 8 * stride + col * 8;
                                idct_islow(&blk, q, &mut c.plane[off..], stride);
                            }
                            kind => {
                                if insufficient {
                                    continue;
                                }
                                let b = (row * c.pbw + col) * 64;
                                let coef = &mut c.coefs[b..b + 64];
                                match kind {
                                    ScanKind::DcFirst => {
                                        let Some(dc) = &dc_t[si] else {
                                            bail!("corrupt JPEG: missing Huffman table");
                                        };
                                        let t = br.huff(dc)? as u32;
                                        let diff = extend(br.get(t), t);
                                        let Some(p) = c.pred.checked_add(diff) else {
                                            bail!("corrupt JPEG: DC coefficient overflow");
                                            // JERR_BAD_DCT_COEF
                                        };
                                        c.pred = p;
                                        coef[0] = ((c.pred as i64) << scan.al) as i16;
                                    }
                                    ScanKind::DcRefine => {
                                        if br.get(1) != 0 {
                                            coef[0] |= (1i32 << scan.al) as i16;
                                        }
                                    }
                                    ScanKind::AcFirst => {
                                        let Some(ac) = &ac_t[si] else {
                                            bail!("corrupt JPEG: missing Huffman table");
                                        };
                                        ac_first(&mut br, ac, scan, &mut eobrun, coef)?;
                                    }
                                    _ => {
                                        let Some(ac) = &ac_t[si] else {
                                            bail!("corrupt JPEG: missing Huffman table");
                                        };
                                        ac_refine(&mut br, ac, scan, &mut eobrun, coef)?;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !insufficient && br.overrun() {
                if br.marker_at.is_none() {
                    bail!("truncated JPEG (entropy-coded data ends early)");
                }
                // libjpeg: "premature end of data segment" -> the rest of this
                // restart interval stays zero.
                insufficient = true;
            }
        }
    }
    Ok(br.marker_at.unwrap_or(br.pos))
}

fn seq_block(
    br: &mut Bits,
    dc: &Huff,
    ac: &Huff,
    pred: &mut i32,
    blk: &mut [i16; 64],
) -> Result<(), ImageError> {
    let t = br.huff(dc)? as u32;
    let diff = extend(br.get(t), t);
    *pred = pred.wrapping_add(diff);
    blk[0] = *pred as i16;
    let mut k = 1usize;
    while k < 64 {
        let rs = br.huff(ac)?;
        let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
        if s != 0 {
            k += r;
            let v = extend(br.get(s), s);
            blk[NATURAL[k]] = v as i16;
        } else {
            if r != 15 {
                break;
            }
            k += 15;
        }
        k += 1;
    }
    Ok(())
}

fn ac_first(
    br: &mut Bits,
    ac: &Huff,
    scan: &Scan,
    eobrun: &mut u32,
    coef: &mut [i16],
) -> Result<(), ImageError> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = scan.ss;
    while k <= scan.se {
        let rs = br.huff(ac)?;
        let (r, s) = ((rs >> 4) as u32, (rs & 15) as u32);
        if s != 0 {
            k += r as usize;
            let v = extend(br.get(s), s);
            coef[NATURAL[k]] = ((v as i64) << scan.al) as i16;
        } else if r == 15 {
            k += 15;
        } else {
            let mut run = 1u32 << r;
            if r != 0 {
                run += br.get(r);
            }
            *eobrun = run - 1;
            break;
        }
        k += 1;
    }
    Ok(())
}

fn ac_refine(
    br: &mut Bits,
    ac: &Huff,
    scan: &Scan,
    eobrun: &mut u32,
    coef: &mut [i16],
) -> Result<(), ImageError> {
    let p1 = (1i32 << scan.al) as i16;
    let m1 = ((-1i32) << scan.al) as i16;
    let se = scan.se;
    let mut k = scan.ss;
    let refine = |br: &mut Bits, c: &mut i16| {
        if br.get(1) != 0 && (*c & p1) == 0 {
            *c = if *c >= 0 {
                c.wrapping_add(p1)
            } else {
                c.wrapping_add(m1)
            };
        }
    };
    if *eobrun == 0 {
        while k <= se {
            let rs = br.huff(ac)?;
            let mut r = (rs >> 4) as i32;
            let s = (rs & 15) as u32;
            let mut newval = 0i16;
            if s != 0 {
                // A size other than 1 is a libjpeg warning, not an error.
                newval = if br.get(1) != 0 { p1 } else { m1 };
            } else if r != 15 {
                let mut run = 1u32 << r;
                if r != 0 {
                    run += br.get(r as u32);
                }
                *eobrun = run;
                break;
            }
            loop {
                let c = &mut coef[NATURAL[k]];
                if *c != 0 {
                    refine(br, c);
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
                if k > se {
                    break;
                }
            }
            if newval != 0 {
                coef[NATURAL[k]] = newval;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            let c = &mut coef[NATURAL[k]];
            if *c != 0 {
                refine(br, c);
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

// ---------------------------------------------------------------- IDCT ----
//
// Pillow's libjpeg-turbo runs `jsimd_idct_islow_avx2` on x86-64. It is the
// ISLOW algorithm of jidctint.c with the same 13-bit constants, factored for
// `vpmaddwd`, so for in-range coefficients it equals the C reference exactly.
// It differs only at the extremes (corrupt or zero-filled data), and we
// reproduce those lane semantics too: dequantization is a wrapping 16-bit
// multiply (`vpmullw`), in0 +/- in4 and in7+in3 / in5+in1 are 16-bit sums
// (`vpaddw`), each pass packs to i16 with signed saturation (`vpackssdw`),
// and output samples saturate (`vpacksswb` + 128) instead of going through
// the C post-IDCT range-limit table.

const CONST_BITS: u32 = 13;
const F_0_298: i32 = 2446;
const F_0_390: i32 = 3196;
const F_0_541: i32 = 4433;
const F_0_765: i32 = 6270;
const F_0_899: i32 = 7373;
const F_1_175: i32 = 9633;
const F_1_501: i32 = 12299;
const F_1_847: i32 = 15137;
const F_1_961: i32 = 16069;
const F_2_053: i32 = 16819;
const F_2_562: i32 = 20995;
const F_3_072: i32 = 25172;

#[inline(always)]
fn sat16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[inline(always)]
fn sample(v: i16) -> u8 {
    (v.clamp(-128, 127) + 128) as u8
}

/// One 8-point pass of the AVX2 `DODCT` macro; `n` = 11 (pass 1) or 18.
#[inline(always)]
fn dodct(x: [i16; 8], n: u32) -> [i16; 8] {
    let i = |k: usize| x[k] as i32;
    // Even part.
    let tmp3 = i(2)
        .wrapping_mul(F_0_541 + F_0_765)
        .wrapping_add(i(6).wrapping_mul(F_0_541));
    let tmp2 = i(6)
        .wrapping_mul(F_0_541 - F_1_847)
        .wrapping_add(i(2).wrapping_mul(F_0_541));
    let tmp0 = (x[0].wrapping_add(x[4]) as i32) << CONST_BITS;
    let tmp1 = (x[0].wrapping_sub(x[4]) as i32) << CONST_BITS;
    let tmp10 = tmp0.wrapping_add(tmp3);
    let tmp13 = tmp0.wrapping_sub(tmp3);
    let tmp11 = tmp1.wrapping_add(tmp2);
    let tmp12 = tmp1.wrapping_sub(tmp2);
    // Odd part.
    let z3 = x[7].wrapping_add(x[3]) as i32;
    let z4 = x[5].wrapping_add(x[1]) as i32;
    let z3p = z3
        .wrapping_mul(F_1_175 - F_1_961)
        .wrapping_add(z4.wrapping_mul(F_1_175));
    let z4p = z4
        .wrapping_mul(F_1_175 - F_0_390)
        .wrapping_add(z3.wrapping_mul(F_1_175));
    let t0 = i(7)
        .wrapping_mul(F_0_298 - F_0_899)
        .wrapping_add(i(1).wrapping_mul(-F_0_899))
        .wrapping_add(z3p);
    let t1 = i(5)
        .wrapping_mul(F_2_053 - F_2_562)
        .wrapping_add(i(3).wrapping_mul(-F_2_562))
        .wrapping_add(z4p);
    let t3 = i(7)
        .wrapping_mul(-F_0_899)
        .wrapping_add(i(1).wrapping_mul(F_1_501 - F_0_899))
        .wrapping_add(z4p);
    let t2 = i(5)
        .wrapping_mul(-F_2_562)
        .wrapping_add(i(3).wrapping_mul(F_3_072 - F_2_562))
        .wrapping_add(z3p);
    let r = 1i32 << (n - 1);
    let d = |v: i32| sat16(v.wrapping_add(r) >> n);
    [
        d(tmp10.wrapping_add(t3)),
        d(tmp11.wrapping_add(t2)),
        d(tmp12.wrapping_add(t1)),
        d(tmp13.wrapping_add(t0)),
        d(tmp13.wrapping_sub(t0)),
        d(tmp12.wrapping_sub(t1)),
        d(tmp11.wrapping_sub(t2)),
        d(tmp10.wrapping_sub(t3)),
    ]
}

/// Pass 2 for one workspace row, written as samples.
#[inline(always)]
fn idct_row(w: &[i16], o: &mut [u8]) {
    if w[1..8].iter().all(|&v| v == 0) {
        // Same value the full pass gives for a lone in0.
        let v = sample(sat16(
            ((w[0] as i32) << CONST_BITS).wrapping_add(1 << 17) >> 18,
        ));
        o[..8].fill(v);
        return;
    }
    let d = dodct([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]], 18);
    for (s, &v) in o[..8].iter_mut().zip(&d) {
        *s = sample(v);
    }
}

/// `jsimd_idct_islow_avx2`: dequantize + ISLOW IDCT of one block into `out`.
fn idct_islow(coef: &[i16; 64], q: &[i32; 64], out: &mut [u8], stride: usize) {
    let mut x = [0i16; 64];
    for (d, (&c, &qq)) in x.iter_mut().zip(coef.iter().zip(q.iter())) {
        *d = (c as i32 * qq) as i16; // vpmullw keeps the low 16 bits
    }
    let mut ws = [0i16; 64];
    if coef[8..].iter().all(|&c| c == 0) {
        // Whole-block shortcut: column c = (in0 << PASS1_BITS) in 16 bits,
        // so every workspace row is identical.
        for c in 0..8 {
            ws[c] = ((x[c] as i32) << 2) as i16;
        }
        let mut row = [0u8; 8];
        idct_row(&ws[..8], &mut row);
        for r in 0..8 {
            out[r * stride..r * stride + 8].copy_from_slice(&row);
        }
        return;
    }
    for c in 0..8 {
        let col = [
            x[c],
            x[8 + c],
            x[16 + c],
            x[24 + c],
            x[32 + c],
            x[40 + c],
            x[48 + c],
            x[56 + c],
        ];
        let d = if col[1..].iter().all(|&v| v == 0) {
            [sat16(4 * col[0] as i32); 8] // exactly what dodct gives for a lone in0
        } else {
            dodct(col, 11)
        };
        for r in 0..8 {
            ws[r * 8 + c] = d[r];
        }
    }
    for r in 0..8 {
        idct_row(&ws[r * 8..r * 8 + 8], &mut out[r * stride..r * stride + 8]);
    }
}

// ------------------------------------------------- upsample + color ----

// ------------------------------------------------ block smoothing ----

/// jdcoefct.c `smoothing_ok`: per component `(coef_bits_latch[0..10],
/// prev_coef_bits_latch[0..10])` when smoothing applies, else None.
fn smoothing_latches(frame: &Frame, scans: usize) -> Option<Vec<([i32; 10], [i32; 10])>> {
    let mut useful = false;
    let mut out = Vec::with_capacity(frame.comps.len());
    for c in &frame.comps {
        c.q?; // every component's quantization table must be latched
        if [0usize, 1, 8, 16, 9, 2, 3, 10, 17, 24]
            .iter()
            .any(|&i| c.qraw[i] == 0)
        {
            return None;
        }
        if c.coef_bits[0] < 0 {
            return None;
        }
        let mut latch = [0i32; 10];
        let mut prev = [0i32; 10];
        latch[0] = c.coef_bits[0];
        for k in 1..10 {
            prev[k] = if scans > 1 { c.prev_coef_bits[k] } else { -1 };
            latch[k] = c.coef_bits[k];
            useful |= c.coef_bits[k] != 0;
        }
        out.push((latch, prev));
    }
    useful.then_some(out)
}

/// jdcoefct.c `decompress_smooth_data` for one component: IDCT every block
/// after estimating still-zero low-frequency coefficients from the 5x5
/// neighbourhood of DC values (row/edge handling reproduced as written,
/// including its use of the current iMCU row's block count).
fn idct_smoothed(
    c: &mut Comp,
    latch: &([i32; 10], [i32; 10]),
    total_rows: usize,
    last_good: usize,
) {
    let q = c.q.unwrap_or([0; 64]);
    let qraw = c.qraw;
    let stride = c.stride();
    let (v, bw, bh, pbw) = (c.v, c.bw, c.bh, c.pbw);
    let last_col = bw as i64 - 1;
    for r in 0..total_rows {
        let block_rows = if r + 1 < total_rows {
            v
        } else {
            match bh % v {
                0 => v,
                k => k,
            }
        };
        let bits = if r > last_good { &latch.1 } else { &latch.0 };
        let change_dc = bits[1..10].iter().all(|&b| b == -1);
        let image_block_rows = (block_rows * total_rows) as i64;
        for b in 0..block_rows {
            let ibr = (r * block_rows + b) as i64;
            let row = r * v + b;
            let prev = if ibr > 0 { row - 1 } else { row };
            let pprev = if ibr > 1 { row - 2 } else { prev };
            let next = if ibr < image_block_rows - 1 {
                row + 1
            } else {
                row
            };
            let nnext = if ibr < image_block_rows - 2 {
                row + 2
            } else {
                next
            };
            let rows = [pprev, prev, row, next, nnext];
            for col in 0..bw {
                let mut d = [0i64; 25];
                for (i, &rr) in rows.iter().enumerate() {
                    for j in 0..5 {
                        let cc = (col as i64 + j as i64 - 2).clamp(0, last_col) as usize;
                        d[i * 5 + j] = c.coefs[(rr * pbw + cc) * 64] as i64;
                    }
                }
                let base = (row * pbw + col) * 64;
                let mut ws = [0i16; 64];
                ws.copy_from_slice(&c.coefs[base..base + 64]);
                smooth_block(&mut ws, &d, bits, change_dc, &qraw);
                idct_islow(&ws, &q, &mut c.plane[row * 8 * stride + col * 8..], stride);
            }
        }
    }
}

/// Coefficient estimates of `decompress_smooth_data`; `d[n - 1]` is DCnn
/// (rows prev-prev..next-next, columns -2..+2).
fn smooth_block(
    ws: &mut [i16; 64],
    d: &[i64; 25],
    bits: &[i32; 10],
    change_dc: bool,
    q: &[u16; 64],
) {
    let dc = |n: usize| d[n - 1];
    let q00 = q[0] as i64;
    let pred = |num: i64, qk: u16, al: i32| -> i16 {
        let qk = qk as i64;
        let mut p = if num >= 0 {
            ((qk << 7) + num) / (qk << 8)
        } else {
            ((qk << 7) - num) / (qk << 8)
        };
        if al > 0 && p >= (1 << al) {
            p = (1 << al) - 1;
        }
        if num < 0 {
            p = -p;
        }
        p as i32 as i16
    };
    // AC01
    if bits[1] != 0 && ws[1] == 0 {
        let s = if change_dc {
            -dc(1) - dc(2) + dc(4) + dc(5) - 3 * dc(6) + 13 * dc(7) - 13 * dc(9) + 3 * dc(10)
                - 3 * dc(11)
                + 38 * dc(12)
                - 38 * dc(14)
                + 3 * dc(15)
                - 3 * dc(16)
                + 13 * dc(17)
                - 13 * dc(19)
                + 3 * dc(20)
                - dc(21)
                - dc(22)
                + dc(24)
                + dc(25)
        } else {
            -7 * dc(11) + 50 * dc(12) - 50 * dc(14) + 7 * dc(15)
        };
        ws[1] = pred(q00 * s, q[1], bits[1]);
    }
    // AC10
    if bits[2] != 0 && ws[8] == 0 {
        let s = if change_dc {
            -dc(1) - 3 * dc(2) - 3 * dc(3) - 3 * dc(4) - dc(5) - dc(6)
                + 13 * dc(7)
                + 38 * dc(8)
                + 13 * dc(9)
                - dc(10)
                + dc(16)
                - 13 * dc(17)
                - 38 * dc(18)
                - 13 * dc(19)
                + dc(20)
                + dc(21)
                + 3 * dc(22)
                + 3 * dc(23)
                + 3 * dc(24)
                + dc(25)
        } else {
            -7 * dc(3) + 50 * dc(8) - 50 * dc(18) + 7 * dc(23)
        };
        ws[8] = pred(q00 * s, q[8], bits[2]);
    }
    // AC20
    if bits[3] != 0 && ws[16] == 0 {
        let s = if change_dc {
            dc(3) + 2 * dc(7) + 7 * dc(8) + 2 * dc(9) - 5 * dc(12) - 14 * dc(13) - 5 * dc(14)
                + 2 * dc(17)
                + 7 * dc(18)
                + 2 * dc(19)
                + dc(23)
        } else {
            -dc(3) + 13 * dc(8) - 24 * dc(13) + 13 * dc(18) - dc(23)
        };
        ws[16] = pred(q00 * s, q[16], bits[3]);
    }
    // AC11
    if bits[4] != 0 && ws[9] == 0 {
        let s = if change_dc {
            -dc(1) + dc(5) + 9 * dc(7) - 9 * dc(9) - 9 * dc(17) + 9 * dc(19) + dc(21) - dc(25)
        } else {
            dc(10) + dc(16) - 10 * dc(17) + 10 * dc(19) - dc(2) - dc(20) + dc(22) - dc(24) + dc(4)
                - dc(6)
                + 10 * dc(7)
                - 10 * dc(9)
        };
        ws[9] = pred(q00 * s, q[9], bits[4]);
    }
    // AC02
    if bits[5] != 0 && ws[2] == 0 {
        let s = if change_dc {
            2 * dc(7) - 5 * dc(8) + 2 * dc(9) + dc(11) + 7 * dc(12) - 14 * dc(13)
                + 7 * dc(14)
                + dc(15)
                + 2 * dc(17)
                - 5 * dc(18)
                + 2 * dc(19)
        } else {
            -dc(11) + 13 * dc(12) - 24 * dc(13) + 13 * dc(14) - dc(15)
        };
        ws[2] = pred(q00 * s, q[2], bits[5]);
    }
    if change_dc {
        // AC03, AC12, AC21, AC30
        if bits[6] != 0 && ws[3] == 0 {
            let s = dc(7) - dc(9) + 2 * dc(12) - 2 * dc(14) + dc(17) - dc(19);
            ws[3] = pred(q00 * s, q[3], bits[6]);
        }
        if bits[7] != 0 && ws[10] == 0 {
            let s = dc(7) - 3 * dc(8) + dc(9) - dc(17) + 3 * dc(18) - dc(19);
            ws[10] = pred(q00 * s, q[10], bits[7]);
        }
        if bits[8] != 0 && ws[17] == 0 {
            let s = dc(7) - dc(9) - 3 * dc(12) + 3 * dc(14) + dc(17) - dc(19);
            ws[17] = pred(q00 * s, q[17], bits[8]);
        }
        if bits[9] != 0 && ws[24] == 0 {
            let s = dc(7) + 2 * dc(8) + dc(9) - dc(17) - 2 * dc(18) - dc(19);
            ws[24] = pred(q00 * s, q[24], bits[9]);
        }
        // DC: Gaussian-like 5x5 average of the DC values (weights sum to 256).
        let s = -2 * dc(1) - 6 * dc(2) - 8 * dc(3) - 6 * dc(4) - 2 * dc(5) - 6 * dc(6)
            + 6 * dc(7)
            + 42 * dc(8)
            + 6 * dc(9)
            - 6 * dc(10)
            - 8 * dc(11)
            + 42 * dc(12)
            + 152 * dc(13)
            + 42 * dc(14)
            - 8 * dc(15)
            - 6 * dc(16)
            + 6 * dc(17)
            + 42 * dc(18)
            + 6 * dc(19)
            - 6 * dc(20)
            - 2 * dc(21)
            - 6 * dc(22)
            - 8 * dc(23)
            - 6 * dc(24)
            - 2 * dc(25);
        ws[0] = pred(q00 * s, q[0], 0);
    }
}

fn finish(frame: &mut Frame, color: Color, scans: usize) -> Result<RgbImage, ImageError> {
    if frame.progressive {
        if let Some(latches) = smoothing_latches(frame, scans) {
            let (t, last_good) = (frame.mcus_y, frame.last_good);
            for (c, latch) in frame.comps.iter_mut().zip(&latches) {
                idct_smoothed(c, latch, t, last_good);
                c.coefs = Vec::new();
            }
        }
        for c in frame.comps.iter_mut().filter(|c| !c.coefs.is_empty()) {
            let q = c.q.unwrap_or([0; 64]);
            let stride = c.stride();
            let mut blk = [0i16; 64];
            for row in 0..c.bh {
                for col in 0..c.bw {
                    let b = (row * c.pbw + col) * 64;
                    blk.copy_from_slice(&c.coefs[b..b + 64]);
                    idct_islow(&blk, &q, &mut c.plane[row * 8 * stride + col * 8..], stride);
                }
            }
            c.coefs = Vec::new();
        }
    } else {
        // A component no scan covered stays all-zero coefficients: mid-gray.
        for c in &mut frame.comps {
            if !c.scanned {
                c.plane.fill(128);
            }
        }
    }
    let (w, h) = (frame.width, frame.height);
    let mut out = vec![0u8; w * h * 3];
    let mut bufs: Vec<Vec<u8>> = frame.comps.iter().map(|_| vec![0u8; w + 16]).collect();
    let mut colsum: Vec<i32> = vec![0; w + 16];
    let yc = ycc_tables();
    for y in 0..h {
        let o = &mut out[y * w * 3..(y + 1) * w * 3];
        match color {
            Color::Gray => {
                let r0 = upsample_row(&frame.comps[0], y, w, &mut bufs[0], &mut colsum);
                for (px, &v) in o.chunks_exact_mut(3).zip(r0) {
                    px.copy_from_slice(&[v, v, v]);
                }
            }
            _ => {
                let (b0, rest) = bufs.split_at_mut(1);
                let (b1, b2) = rest.split_at_mut(1);
                let r0 = upsample_row(&frame.comps[0], y, w, &mut b0[0], &mut colsum);
                let r1 = upsample_row(&frame.comps[1], y, w, &mut b1[0], &mut colsum);
                let r2 = upsample_row(&frame.comps[2], y, w, &mut b2[0], &mut colsum);
                if color == Color::Rgb {
                    for (x, px) in o.chunks_exact_mut(3).enumerate() {
                        px.copy_from_slice(&[r0[x], r1[x], r2[x]]);
                    }
                } else {
                    for (x, px) in o.chunks_exact_mut(3).enumerate() {
                        let yy = r0[x] as i32;
                        let (cb, cr) = (r1[x] as usize, r2[x] as usize);
                        px[0] = clamp8(yy + yc.cr_r[cr]);
                        px[1] = clamp8(yy + ((yc.cb_g[cb] + yc.cr_g[cr]) >> 16));
                        px[2] = clamp8(yy + yc.cb_b[cb]);
                    }
                }
            }
        }
    }
    Ok(RgbImage {
        width: w as u32,
        height: h as u32,
        data: out,
    })
}

#[inline(always)]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Output row `y` of component `c`, upsampled to at least `w` samples.
fn upsample_row<'b>(
    c: &'b Comp,
    y: usize,
    w: usize,
    buf: &'b mut [u8],
    colsum: &mut [i32],
) -> &'b [u8] {
    let stride = c.stride();
    let plane = &c.plane;
    let row = |r: usize| &plane[r * stride..r * stride + c.dw];
    let last = c.dh - 1;
    match c.up {
        Up::Full => &plane[y * stride..y * stride + w],
        Up::H2V1Fancy => {
            let src = row(y);
            h2v1_fancy(src, buf);
            &buf[..w]
        }
        Up::H1V2Fancy => {
            let i = y / 2;
            let (far, bias) = if y.is_multiple_of(2) {
                (i.saturating_sub(1), 1)
            } else {
                ((i + 1).min(last), 2)
            };
            let (near, far) = (row(i), row(far));
            for x in 0..c.dw {
                buf[x] = ((3 * near[x] as i32 + far[x] as i32 + bias) >> 2) as u8;
            }
            &buf[..w]
        }
        Up::H2V2Fancy => {
            let i = y / 2;
            let far = if y.is_multiple_of(2) {
                i.saturating_sub(1)
            } else {
                (i + 1).min(last)
            };
            let (near, far) = (row(i), row(far));
            let n = c.dw;
            for x in 0..n {
                colsum[x] = 3 * near[x] as i32 + far[x] as i32;
            }
            for x in 0..n {
                let this = colsum[x];
                let prev = colsum[x.saturating_sub(1)];
                let next = colsum[(x + 1).min(n - 1)];
                buf[2 * x] = ((this * 3 + prev + 8) >> 4) as u8;
                buf[2 * x + 1] = ((this * 3 + next + 7) >> 4) as u8;
            }
            &buf[..w]
        }
        Up::Box(he, ve) => {
            let src = row(y / ve);
            for (x, d) in buf[..w].iter_mut().enumerate() {
                *d = src[x / he];
            }
            &buf[..w]
        }
    }
}

fn h2v1_fancy(src: &[u8], out: &mut [u8]) {
    let n = src.len();
    for x in 0..n {
        let this = 3 * src[x] as i32;
        let prev = src[x.saturating_sub(1)] as i32;
        let next = src[(x + 1).min(n - 1)] as i32;
        out[2 * x] = ((this + prev + 1) >> 2) as u8;
        out[2 * x + 1] = ((this + next + 2) >> 2) as u8;
    }
}

struct Ycc {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

/// `build_ycc_rgb_table` (jdcolor.c, SCALEBITS 16).
fn ycc_tables() -> &'static Ycc {
    static T: OnceLock<Ycc> = OnceLock::new();
    T.get_or_init(|| {
        const ONE_HALF: i64 = 1 << 15;
        let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
        let mut t = Ycc {
            cr_r: [0; 256],
            cb_b: [0; 256],
            cr_g: [0; 256],
            cb_g: [0; 256],
        };
        for i in 0..256usize {
            let x = i as i64 - 128;
            t.cr_r[i] = ((fix(1.40200) * x + ONE_HALF) >> 16) as i32;
            t.cb_b[i] = ((fix(1.77200) * x + ONE_HALF) >> 16) as i32;
            t.cr_g[i] = (-fix(0.71414) * x) as i32;
            t.cb_g[i] = (-fix(0.34414) * x + ONE_HALF) as i32;
        }
        t
    })
}

/// JPEG Annex K.3 tables that libjpeg-turbo installs into empty slots 0/1.
fn std_tables() -> [(usize, usize, &'static [u8], &'static [u8]); 4] {
    const DC_LUM_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    const DC_CHR_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    const AC_LUM_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
    const AC_LUM_VALS: [u8; 162] = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
        0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
        0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
        0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
        0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
        0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
        0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
        0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];
    const AC_CHR_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
    const AC_CHR_VALS: [u8; 162] = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
        0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
        0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
        0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
        0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
        0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
        0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
        0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
        0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
        0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];
    [
        (0, 0, &DC_LUM_BITS, &DC_VALS),
        (1, 0, &AC_LUM_BITS, &AC_LUM_VALS),
        (0, 1, &DC_CHR_BITS, &DC_VALS),
        (1, 1, &AC_CHR_BITS, &AC_CHR_VALS),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycc_table_values() {
        let t = ycc_tables();
        // FIX(1.402) = 91881, FIX(1.772) = 116130, FIX(0.71414) = 46802,
        // FIX(0.34414) = 22554.
        assert_eq!(t.cr_r[255], ((91881i64 * 127 + 32768) >> 16) as i32);
        assert_eq!(t.cb_b[0], ((116130i64 * -128 + 32768) >> 16) as i32);
        assert_eq!(t.cr_g[0], 46802 * 128);
        assert_eq!(t.cb_g[128], 32768);
    }

    #[test]
    fn idct_saturates_like_avx2() {
        // A huge DC saturates to 255 (the C range-limit table would wrap).
        let mut coef = [0i16; 64];
        coef[0] = 2000;
        let q = [16i32; 64];
        let mut out = [0u8; 64];
        idct_islow(&coef, &q, &mut out, 8);
        // 2000*16 = 32000 fits i16; << 2 wraps in 16 bits: 128000 & 0xffff
        // = -3072 -> (-3072 << 13 + 2^17) >> 18 = -96 -> sample 32.
        assert!(out.iter().all(|&v| v == 32), "{out:?}");
        coef[1] = 1; // defeats the whole-block shortcut; row 0 keeps AC
        coef[9] = 1; // row 1 nonzero: full column path saturates instead
        idct_islow(&coef, &q, &mut out, 8);
        assert!(out.iter().all(|&v| v == 255), "{out:?}");
    }

    #[test]
    fn idct_dc_only_is_flat() {
        let mut coef = [0i16; 64];
        coef[0] = 10;
        let q = [16i32; 64];
        let mut out = [0u8; 64];
        idct_islow(&coef, &q, &mut out, 8);
        // DC 10*16=160 -> pass 1 *4 = 640 -> (640 << 13 + 2^17) >> 18 = 20 -> 148
        assert!(out.iter().all(|&v| v == 148), "{out:?}");
    }

    #[test]
    fn std_tables_are_valid_codes() {
        for (class, _, bits, vals) in std_tables() {
            let mut b = [0u8; 17];
            b[1..].copy_from_slice(bits);
            let mut v = [0u8; 256];
            v[..vals.len()].copy_from_slice(vals);
            assert_eq!(bits.iter().map(|&x| x as usize).sum::<usize>(), vals.len());
            Huff::new(&RawHuff { bits: b, vals: v }, class == 0).unwrap();
        }
    }
}
