//! Token ids and the MiMo special-token strings (A2 F1). T29 hygiene: every
//! special string is built from parts (`\x3c`/`\x3e`/`|` escapes) — the literal
//! tags never appear in this source. Ids are from `generation_config.json` /
//! `oracle/mimo26/config.py`.

/// `bos_token_id` / `pad_token_id` (151643).
pub const BOS_TOKEN_ID: u32 = 151_643;
/// The served EOS set (T28: honour all three; greedy stops on the first).
pub const EOS_TOKEN_IDS: [u32; 3] = [151_643, 151_645, 151_672];
pub const PAD_TOKEN_ID: u32 = 151_643;
/// DFlash mask embedding token.
pub const MASK_TOKEN_ID: u32 = 151_675;

/// `<|im_start|>` — the role marker (built from parts).
pub const IM_START: &str = concat!("\x3c", "|im_start|", "\x3e");
/// `<|im_end|>` — role end / EOS marker (built from parts).
pub const IM_END: &str = concat!("\x3c", "|im_end|", "\x3e");
/// `<think>` / `</think>` — reasoning delimiters (built from parts).
pub const THINK: &str = concat!("\x3c", "think", "\x3e");
pub const THINK_CLOSE: &str = concat!("\x3c", "/think", "\x3e");

// Multimodal pads (the served template references them; v1 answers media with
// 400 until encoders land — they are rendered only if a media part is passed,
// which v1 does not allow).
pub const VISION_START: &str = concat!("\x3c", "|vision_start|", "\x3e");
pub const IMAGE_PAD: &str = concat!("\x3c", "|image_pad|", "\x3e");
pub const VISION_END: &str = concat!("\x3c", "|vision_end|", "\x3e");
pub const AUDIO_START: &str = concat!("\x3c", "|mimo_audio_start|", "\x3e");
pub const AUDIO_PAD: &str = concat!("\x3c", "|audio_pad|", "\x3e");
pub const AUDIO_END: &str = concat!("\x3c", "|mimo_audio_end|", "\x3e");
pub const VIDEO_PAD: &str = concat!("\x3c", "|video_pad|", "\x3e");

/// A special token id set the decode loop must treat as end-of-generation.
pub fn is_eos(id: u32) -> bool {
    EOS_TOKEN_IDS.contains(&id)
}
