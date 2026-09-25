//! Vision input for the MiMo-V2.6-Flash engine: an OpenAI chat `image_url` data
//! URL becomes the `pixel_values` the model's HF image processor would produce.
//!
//! Pipeline (each stage is public so it can be tested on its own):
//!
//! 1. [`decode_data_url`] — `data:image/<type>;base64,<payload>` → bytes.
//! 2. [`decode`] — PNG or JPEG (sniffed by magic) → RGB8, matching what HF gets
//!    from Pillow: `Image.open` → `ImageOps.exif_transpose` →
//!    `transformers.image_transforms.convert_to_rgb` (non-RGB modes are
//!    converted to RGBA and alpha-composited over opaque white).
//! 3. [`smart_resize`] — the Qwen2-VL target size rule (factor 32, min/max
//!    pixels), with Python's float semantics.
//! 4. [`resize_bicubic`] — Pillow's `Image.resize(.., BICUBIC)` (Resample.c).
//! 5. [`preprocess`] — rescale + normalize (CLIP mean/std, f32, exactly the
//!    numpy expression HF evaluates) and the Qwen2-VL patch layout.
//!
//! No third-party crates and no `unsafe`. Decoders never panic on hostile
//! input; they return [`ImageError`] (whose `Display` is safe for an HTTP 400
//! body). Images above [`MAX_DECODE_PIXELS`] are rejected from the header,
//! before any pixel buffer is allocated.
//!
//! Fidelity: bit-exact against Pillow 12.2 (libjpeg-turbo 3.1, x86-64 AVX2
//! IDCT) and numpy on every golden in `tests/goldens` (see
//! `tests/gen_goldens.py`). Deliberate differences from Pillow, all in
//! error/edge cases: CMYK/YCCK, arithmetic, lossless, hierarchical and
//! non-8-bit JPEGs are rejected (libjpeg-turbo would decode some); PNG CRCs
//! are not verified; a malformed Exif block is ignored where Pillow's
//! `exif_transpose` would raise; PNG XMP and "Raw profile type exif" text
//! chunks are not consulted for orientation (PNG `eXIf` is).
//!
//! Note that `transformers.image_utils.load_image` (URL / base64 strings)
//! ends with `image.convert("RGB")`, which drops alpha instead of compositing
//! over white; this crate follows the image processor's `convert_to_rgb`.
#![forbid(unsafe_code)]

mod base64;
mod exif;
mod inflate;
mod jpeg;
mod png;
mod preprocess;
mod resize;

use std::fmt;

pub use preprocess::normalize_value;

/// Vision patch size (`patch_size`).
pub const PATCH: usize = 16;
/// Spatial merge size (`merge_size` / `spatial_merge_size`).
pub const MERGE: usize = 2;
/// Temporal patch size (`temporal_patch_size`); a still image is duplicated.
pub const TEMPORAL: usize = 2;
/// `smart_resize` factor: `PATCH * MERGE`.
pub const FACTOR: u32 = 32;
/// `min_pixels` (`size.shortest_edge`) of the checkpoint's preprocessor_config.
pub const MIN_PIXELS: u64 = 3136;
/// `max_pixels` (`size.longest_edge`) of the checkpoint's preprocessor_config.
pub const MAX_PIXELS: u64 = 12_845_056;
/// Floats per patch row: `[channel 3][temporal 2][py 16][px 16]`.
pub const PATCH_DIM: usize = 3 * TEMPORAL * PATCH * PATCH; // 1536
/// Decoder guard: images with `width * height` above this (64 Mi pixels,
/// e.g. 8192x8192) are rejected from the header before decoding.
pub const MAX_DECODE_PIXELS: u64 = 64 << 20;
/// CLIP normalization mean (preprocessor_config `image_mean`), as the f64
/// Python floats; HF casts them to float32 (`np.array(mean, dtype=float32)`).
pub const IMAGE_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
/// CLIP normalization std (preprocessor_config `image_std`).
pub const IMAGE_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

// The workspace builds on 64-bit hosts; pixel-count arithmetic below relies on
// `usize` holding any `u32 * u32 * 4`.
const _: () = assert!(std::mem::size_of::<usize>() == 8);

/// A human-readable decode/preprocess failure (safe to return in an HTTP 400 body).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageError(pub String);

impl ImageError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        ImageError(msg.into())
    }
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ImageError {}

/// Shorthand for building an `Err(ImageError)` with `format!` arguments.
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::ImageError::new(format!($($arg)*)))
    };
}
pub(crate) use bail;

/// An 8-bit RGB image, row-major, `data.len() == width * height * 3`.
#[derive(Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl fmt::Debug for RgbImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RgbImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("data.len()", &self.data.len())
            .finish()
    }
}

/// `pixel_values` for one image: `data` is `[grid_t * grid_h * grid_w, PATCH_DIM]`
/// row-major f32. Rows are ordered by merged 2x2 unit (row-major over
/// `(grid_h / 2, grid_w / 2)`), then the unit's four patches in `(dy, dx)`
/// order; each row is `[channel 3][temporal 2][py 16][px 16]` with the same
/// frame in both temporal slots.
#[derive(Clone, PartialEq)]
pub struct Patches {
    pub grid_t: u32,
    pub grid_h: u32,
    pub grid_w: u32,
    pub data: Vec<f32>,
}

impl Patches {
    /// Number of patch rows (`grid_t * grid_h * grid_w`).
    pub fn rows(&self) -> usize {
        self.grid_t as usize * self.grid_h as usize * self.grid_w as usize
    }

    /// LLM tokens after the 2x2 spatial merge.
    pub fn merged_tokens(&self) -> usize {
        self.rows() / (MERGE * MERGE)
    }
}

impl fmt::Debug for Patches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Patches")
            .field("grid_t", &self.grid_t)
            .field("grid_h", &self.grid_h)
            .field("grid_w", &self.grid_w)
            .field("data.len()", &self.data.len())
            .finish()
    }
}

/// Decode an OpenAI `image_url` data URL (`data:image/<any>;base64,<b64>`).
/// ASCII whitespace inside the base64 payload is ignored.
pub fn decode_data_url(url: &str) -> Result<RgbImage, ImageError> {
    let bytes = data_url_bytes(url)?;
    decode(&bytes)
}

/// The raw (still encoded) image bytes carried by a base64 data URL.
pub fn data_url_bytes(url: &str) -> Result<Vec<u8>, ImageError> {
    let s = url.trim_matches(|c: char| c.is_ascii_whitespace());
    let has_prefix = s.len() >= 5 && s.as_bytes()[..5].eq_ignore_ascii_case(b"data:");
    if !has_prefix {
        let lower: String = s.chars().take(8).collect::<String>().to_ascii_lowercase();
        if lower.starts_with("http://") || lower.starts_with("https://") {
            bail!(
                "remote image URLs are not supported; send the image inline as \
                 data:image/<type>;base64,<data>"
            );
        }
        bail!(
            "image_url must be a base64 data URL (data:image/<type>;base64,<data>), got {:?}",
            snippet(s)
        );
    }
    let rest = &s[5..];
    let Some(comma) = rest.find(',') else {
        bail!("malformed data URL: no ',' between header and data");
    };
    let header = &rest[..comma];
    let payload = &rest[comma + 1..];
    let mut parts = header.split(';');
    let media = parts.next().unwrap_or("").trim();
    let params: Vec<&str> = parts.map(str::trim).collect();
    let is_base64 = params
        .last()
        .is_some_and(|p| p.eq_ignore_ascii_case("base64"));
    if !is_base64 {
        bail!(
            "data URL is not base64-encoded (expected data:image/<type>;base64,<data>), header {:?}",
            snippet(header)
        );
    }
    let media_l = media.to_ascii_lowercase();
    if !(media_l.is_empty()
        || media_l.starts_with("image/")
        || media_l == "application/octet-stream")
    {
        bail!(
            "data URL media type {:?} is not an image type (expected image/*)",
            snippet(media)
        );
    }
    let bytes = base64::decode(payload.as_bytes())?;
    if bytes.is_empty() {
        bail!("data URL carries no image data");
    }
    Ok(bytes)
}

fn snippet(s: &str) -> String {
    const MAX: usize = 48;
    if s.chars().count() <= MAX {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(MAX).collect();
        t.push('…');
        t
    }
}

/// Decode PNG or JPEG bytes (sniffed by magic, not by the declared media
/// type) to RGB8, applying EXIF orientation and HF `convert_to_rgb`.
pub fn decode(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    if bytes.starts_with(&png::SIGNATURE) {
        png::decode(bytes)
    } else if bytes.len() >= 3 && bytes[..3] == [0xFF, 0xD8, 0xFF] {
        jpeg::decode(bytes)
    } else {
        let what = sniff_other(bytes);
        if what.is_empty() {
            bail!("unsupported image format (PNG and JPEG are supported)");
        }
        bail!("unsupported image format (PNG and JPEG are supported); got {what}");
    }
}

fn sniff_other(b: &[u8]) -> &'static str {
    if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        "GIF"
    } else if b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        "WebP"
    } else if b.starts_with(b"BM") {
        "BMP"
    } else if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") {
        "TIFF"
    } else if b.len() >= 12 && &b[4..8] == b"ftyp" {
        match &b[8..12] {
            b"avif" | b"avis" => "AVIF",
            b"heic" | b"heix" | b"mif1" | b"msf1" | b"hevc" => "HEIF",
            _ => "",
        }
    } else if b.starts_with(&[0xFF, 0xD8]) {
        "(JPEG without a marker after SOI)"
    } else {
        ""
    }
}

/// Qwen2-VL `smart_resize(height, width, factor=32, min_pixels=3136,
/// max_pixels=12845056)`; returns `(height, width)`.
///
/// Mirrors the Python exactly: `round` is round-half-to-even, `int / int` is a
/// correctly rounded division, and the float steps use the same operation
/// order. Errors on a zero side or an aspect ratio above 200.
pub fn smart_resize(height: u32, width: u32) -> Result<(u32, u32), ImageError> {
    if height == 0 || width == 0 {
        bail!("image has zero width or height ({width}x{height})");
    }
    let (h, w) = (height as u64, width as u64);
    let factor = FACTOR as u64;
    let ratio = py_true_div(h.max(w), h.min(w));
    if ratio > 200.0 {
        bail!("absolute aspect ratio must be smaller than 200, got {ratio} ({width}x{height})");
    }
    let fh = h as f64;
    let fw = w as f64;
    let ff = factor as f64;
    let mut h_bar = (fh / ff).round_ties_even() as u64 * factor;
    let mut w_bar = (fw / ff).round_ties_even() as u64 * factor;
    let area = h_bar as u128 * w_bar as u128;
    if area > MAX_PIXELS as u128 {
        let beta = py_true_div(h * w, MAX_PIXELS).sqrt();
        h_bar = factor.max((fh / beta / ff).floor() as u64 * factor);
        w_bar = factor.max((fw / beta / ff).floor() as u64 * factor);
    } else if area < MIN_PIXELS as u128 {
        let beta = py_true_div(MIN_PIXELS, h * w).sqrt();
        h_bar = (fh * beta / ff).ceil() as u64 * factor;
        w_bar = (fw * beta / ff).ceil() as u64 * factor;
    }
    match (u32::try_from(h_bar), u32::try_from(w_bar)) {
        (Ok(a), Ok(b)) => Ok((a, b)),
        _ => bail!("smart_resize result out of range for {width}x{height}"),
    }
}

/// Python's `int / int` for non-negative ints: the correctly rounded
/// (ties-to-even) f64 of the exact quotient. `b > 0`.
fn py_true_div(a: u64, b: u64) -> f64 {
    const EXACT: u64 = 1 << 53;
    if a <= EXACT && b <= EXACT {
        // Both convert exactly; IEEE division is correctly rounded.
        return a as f64 / b as f64;
    }
    if a == 0 {
        return 0.0;
    }
    // Scale the dividend so the integer quotient has >= 55 significant bits,
    // round to 53 bits (ties to even, remainder as sticky bit), rescale.
    let bits = |x: u128| 128 - x.leading_zeros() as i32;
    let (a, b) = (a as u128, b as u128);
    let shift = (55 + bits(b) - bits(a)).max(0); // bits(a) + shift <= 119
    let num = a << shift;
    let (q, r) = (num / b, num % b);
    let drop = bits(q) - 53; // >= 2
    let low = q & ((1u128 << drop) - 1);
    let half = 1u128 << (drop - 1);
    let mut m = q >> drop;
    if low > half || (low == half && (r != 0 || m & 1 == 1)) {
        m += 1;
    }
    let mut exp = drop - shift;
    if m == 1u128 << 53 {
        m >>= 1;
        exp += 1;
    }
    (m as f64) * 2f64.powi(exp)
}

/// LLM tokens for an image of this ORIGINAL size: `(rh / 32) * (rw / 32)`
/// after [`smart_resize`].
pub fn merged_tokens(height: u32, width: u32) -> Result<usize, ImageError> {
    let (rh, rw) = smart_resize(height, width)?;
    Ok((rh / FACTOR) as usize * (rw / FACTOR) as usize)
}

/// Pillow `Image.resize((width, height), Image.BICUBIC)` on an RGB image
/// (Resample.c: a = -0.5, support 2 scaled by the downscale factor, 22-bit
/// fixed-point coefficients, horizontal then vertical pass). Returns a copy
/// when the size is unchanged; a zero target side gives an empty image.
pub fn resize_bicubic(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    resize::resize_bicubic(img, width, height)
}

/// Full HF image-processor path for one RGB image: [`smart_resize`],
/// [`resize_bicubic`] (skipped when the size is unchanged), rescale/normalize
/// ([`normalize_value`]) and the Qwen2-VL patch layout (see [`Patches`]).
pub fn preprocess(img: &RgbImage) -> Result<Patches, ImageError> {
    preprocess::preprocess(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_true_div_matches_python() {
        // (a, b, f64 bits of Python's a / b), printed by CPython 3.12.
        let cases: &[(u64, u64, u64)] = &[
            (18446744073709551615, 12845056, 0x4274e5e0a72f0539),
            (1152921504606859321, 12845056, 0x4234e5e0a72f0578),
            (18446744065119617025, 12845056, 0x4274e5e0a7053978),
            (3136, 18446744073709551615, 0x3ca8800000000000),
            (9007199254740993, 1, 0x4340000000000000),
            (18014398509481986, 1, 0x4350000000000000),
            (18014398509481990, 1, 0x4350000000000002),
            (10000000000000000007, 3, 0x43c7213080c1a6ab),
            (
                18446744073709551615,
                18446744073709551613,
                0x3ff0000000000000,
            ),
            (12345678901234567, 12845056, 0x41cca4c961947005),
        ];
        for &(a, b, want) in cases {
            let got = py_true_div(a, b);
            assert_eq!(got.to_bits(), want, "{a} / {b}: got {got:e}");
        }
        assert_eq!(py_true_div(1, 3), 1.0 / 3.0);
    }

    #[test]
    fn data_url_errors_are_clear() {
        let e = data_url_bytes("https://example.com/x.png").unwrap_err();
        assert!(e.0.contains("remote image URLs are not supported"), "{e}");
        let e = data_url_bytes("hello").unwrap_err();
        assert!(e.0.contains("must be a base64 data URL"), "{e}");
        let e = data_url_bytes("data:image/png,rawbytes").unwrap_err();
        assert!(e.0.contains("not base64-encoded"), "{e}");
        let e = data_url_bytes("data:text/plain;base64,aGVsbG8=").unwrap_err();
        assert!(e.0.contains("not an image type"), "{e}");
        let e = data_url_bytes("data:image/png;base64").unwrap_err();
        assert!(e.0.contains("no ','"), "{e}");
        let e = data_url_bytes("data:image/png;base64,").unwrap_err();
        assert!(e.0.contains("no image data"), "{e}");
        let e = data_url_bytes("data:image/png;base64,iVBO*w0K").unwrap_err();
        assert!(e.0.contains("invalid base64"), "{e}");
    }

    #[test]
    fn data_url_accepts_whitespace_and_case() {
        let b = data_url_bytes("  DATA:Image/PNG;BASE64,aGVs\nbG8g\r\n d29y bGQ=  ").unwrap();
        assert_eq!(b, b"hello world");
        let b = data_url_bytes("data:image/jpeg;name=x.jpg;base64,aGk").unwrap();
        assert_eq!(b, b"hi");
    }

    #[test]
    fn decode_rejects_unknown_formats() {
        let e = decode(b"GIF89a....").unwrap_err();
        assert_eq!(
            e.0,
            "unsupported image format (PNG and JPEG are supported); got GIF"
        );
        let e = decode(b"\x00\x01\x02").unwrap_err();
        assert_eq!(e.0, "unsupported image format (PNG and JPEG are supported)");
        let e = decode(b"").unwrap_err();
        assert_eq!(e.0, "unsupported image format (PNG and JPEG are supported)");
    }

    #[test]
    fn smart_resize_spot_checks() {
        // Values computed with the reference Python (see tests/gen_goldens.py).
        assert_eq!(smart_resize(480, 640).unwrap(), (480, 640));
        assert_eq!(smart_resize(1, 1).unwrap(), (64, 64));
        assert_eq!(smart_resize(16, 16).unwrap(), (64, 64));
        assert_eq!(smart_resize(48, 48).unwrap(), (64, 64));
        assert!(smart_resize(1, 201).is_err());
        assert!(smart_resize(0, 5).is_err());
        assert_eq!(merged_tokens(480, 640).unwrap(), 15 * 20);
    }
}
