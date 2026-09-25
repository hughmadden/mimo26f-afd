//! The sha manifest: every slice file + its sha256 + the source tensor name +
//! the geometry, written next to the slices.
//!
//! Format: JSON, hand-rolled (no serde) so the crate stays dependency-free and
//! the exact bytes are pinned by `tests/manifest.rs`. The parser is strict —
//! an unknown field, a missing field, a malformed sha or a geometry that
//! disagrees with the compiled-in constants is a loud error, never a default.
//!
//! ```json
//! {
//!   "format": "mimo26-repack-manifest",
//!   "version": 2,
//!   "generated_by": "mimo26-repack 0.1.0",
//!   "geometry": {
//!     "hidden": 4096, "intermediate": 2048, "experts_per_layer": 256,
//!     "moe_layers": 47, "ep_ranks": 4,
//!     "expert_bytes": 13369344, "quarter_slice_bytes": 3342336
//!   },
//!   "layout": [
//!     {"proj": "gate_proj", "region": "payload", "off": 0, "len": 1048576},
//!     ...
//!   ],
//!   "slices": [
//!     {
//!       "file": "L01_E000_R0.slice",
//!       "sha256": "…64 hex…",
//!       "bytes": 3342336,
//!       "layer": 1, "expert": 0, "rank": 0,
//!       "source": {
//!         "shard": "model_pp0_ep0_shard0.safetensors",
//!         "tensors": [
//!           {"name": "model.layers.1.mlp.experts.0.gate_proj.weight",
//!            "shape": [2048, 2048], "dtype": "U8"},
//!           ...
//!         ]
//!       }
//!     }
//!   ]
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::error::{io_err, RepackError};
use crate::geom::{self, Proj};
use crate::sha256;

/// Manifest format tag.
pub const FORMAT: &str = "mimo26-repack-manifest";
/// Manifest format version.
pub const VERSION: u32 = 2;

/// One source tensor reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor {
    /// Full checkpoint tensor name.
    pub name: String,
    /// Declared shape.
    pub shape: Vec<usize>,
    /// Declared dtype (always `U8` for expert weights).
    pub dtype: String,
}

/// One slice file's manifest entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceEntry {
    /// File name, relative to the manifest's directory.
    pub file: String,
    /// Lowercase hex sha256 of the file's bytes.
    pub sha256: String,
    /// File length in bytes.
    pub bytes: u64,
    /// MoE layer id.
    pub layer: usize,
    /// Expert id.
    pub expert: usize,
    /// EP rank (quarter slice index).
    pub rank: usize,
    /// Checkpoint shard file the tensors came from.
    pub shard: String,
    /// The six source tensors, in `gate, up, down` x `weight, weight_scale` order.
    pub tensors: Vec<SourceTensor>,
}

/// The whole manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Format tag (must equal [`FORMAT`]).
    pub format: String,
    /// Format version (must equal [`VERSION`]).
    pub version: u32,
    /// Producer string.
    pub generated_by: String,
    /// Geometry block, field name -> value.
    pub geometry: BTreeMap<String, u64>,
    /// Pinned layout table.
    pub layout: Vec<LayoutEntry>,
    /// Slice entries, in file-name order.
    pub slices: Vec<SliceEntry>,
}

/// One row of the manifest's layout table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutEntry {
    /// Projection name.
    pub proj: String,
    /// `payload` | `scales`.
    pub region: String,
    /// Byte offset in the slice.
    pub off: u64,
    /// Byte length.
    pub len: u64,
}

impl Manifest {
    /// A manifest with the compiled-in geometry and layout, no slices yet.
    pub fn new() -> Self {
        let mut geometry = BTreeMap::new();
        geometry.insert("hidden".to_string(), geom::HIDDEN as u64);
        geometry.insert("intermediate".to_string(), geom::INTERMEDIATE as u64);
        geometry.insert(
            "experts_per_layer".to_string(),
            geom::EXPERTS_PER_LAYER as u64,
        );
        geometry.insert("moe_layers".to_string(), geom::MOE_LAYERS as u64);
        geometry.insert("ep_ranks".to_string(), geom::EP_RANKS as u64);
        geometry.insert("expert_bytes".to_string(), geom::EXPERT_BYTES as u64);
        geometry.insert(
            "quarter_slice_bytes".to_string(),
            geom::QUARTER_SLICE_BYTES as u64,
        );
        let layout = geom::layout_table()
            .into_iter()
            .map(|r| LayoutEntry {
                proj: r.proj.name().to_string(),
                region: r.region.to_string(),
                off: r.off as u64,
                len: r.len as u64,
            })
            .collect();
        Manifest {
            format: FORMAT.to_string(),
            version: VERSION,
            generated_by: format!("mimo26-repack {}", env!("CARGO_PKG_VERSION")),
            geometry,
            layout,
            slices: Vec::new(),
        }
    }

    /// Add a slice entry (kept sorted by file name).
    pub fn push(&mut self, entry: SliceEntry) {
        self.slices.push(entry);
        self.slices.sort_by(|a, b| a.file.cmp(&b.file));
    }

    /// Look up a slice entry by file name.
    pub fn get(&self, file: &str) -> Option<&SliceEntry> {
        self.slices.iter().find(|s| s.file == file)
    }

    /// Serialize to the pinned JSON form (2-space indent, one field per line).
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        let _ = writeln!(s, "  \"format\": {},", json_str(&self.format));
        let _ = writeln!(s, "  \"version\": {},", self.version);
        let _ = writeln!(s, "  \"generated_by\": {},", json_str(&self.generated_by));
        s.push_str("  \"geometry\": {\n");
        let n = self.geometry.len();
        for (i, (k, v)) in self.geometry.iter().enumerate() {
            let _ = writeln!(
                s,
                "    {}: {}{}",
                json_str(k),
                v,
                if i + 1 == n { "" } else { "," }
            );
        }
        s.push_str("  },\n");
        s.push_str("  \"layout\": [\n");
        for (i, r) in self.layout.iter().enumerate() {
            let _ = writeln!(
                s,
                "    {{\"proj\": {}, \"region\": {}, \"off\": {}, \"len\": {}}}{}",
                json_str(&r.proj),
                json_str(&r.region),
                r.off,
                r.len,
                if i + 1 == self.layout.len() { "" } else { "," }
            );
        }
        s.push_str("  ],\n");
        s.push_str("  \"slices\": [\n");
        for (i, e) in self.slices.iter().enumerate() {
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"file\": {},", json_str(&e.file));
            let _ = writeln!(s, "      \"sha256\": {},", json_str(&e.sha256));
            let _ = writeln!(s, "      \"bytes\": {},", e.bytes);
            let _ = writeln!(s, "      \"layer\": {},", e.layer);
            let _ = writeln!(s, "      \"expert\": {},", e.expert);
            let _ = writeln!(s, "      \"rank\": {},", e.rank);
            s.push_str("      \"source\": {\n");
            let _ = writeln!(s, "        \"shard\": {},", json_str(&e.shard));
            s.push_str("        \"tensors\": [\n");
            for (j, t) in e.tensors.iter().enumerate() {
                let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
                let _ = writeln!(
                    s,
                    "          {{\"name\": {}, \"shape\": [{}], \"dtype\": {}}}{}",
                    json_str(&t.name),
                    shape.join(", "),
                    json_str(&t.dtype),
                    if j + 1 == e.tensors.len() { "" } else { "," }
                );
            }
            s.push_str("        ]\n");
            s.push_str("      }\n");
            let _ = write!(s, "    }}{}", if i + 1 == self.slices.len() { "" } else { "," });
            s.push('\n');
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        s
    }

    /// Write the manifest to `path`.
    pub fn write(&self, path: &Path) -> Result<(), RepackError> {
        let p = path.display().to_string();
        std::fs::write(path, self.to_json()).map_err(|e| io_err(&p, e))
    }

    /// Read and validate a manifest from `path`.
    pub fn read(path: &Path) -> Result<Self, RepackError> {
        let p = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|e| io_err(&p, e))?;
        Self::parse(&text)
    }

    /// Parse and validate a manifest from JSON text.
    pub fn parse(text: &str) -> Result<Self, RepackError> {
        let v = crate::json::parse(text)?;
        let obj = v
            .as_object()
            .ok_or_else(|| RepackError::BadManifest("top level is not an object".into()))?;
        let format = obj
            .get("format")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RepackError::BadManifest("missing \"format\"".into()))?
            .to_string();
        if format != FORMAT {
            return Err(RepackError::BadManifest(format!(
                "format {format:?} != {FORMAT:?}"
            )));
        }
        let version = obj
            .get("version")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RepackError::BadManifest("missing \"version\"".into()))?;
        if version != VERSION as u64 {
            return Err(RepackError::BadManifest(format!(
                "version {version} != {VERSION}"
            )));
        }
        let generated_by = obj
            .get("generated_by")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let geometry = obj
            .get("geometry")
            .and_then(|v| v.as_object())
            .ok_or_else(|| RepackError::BadManifest("missing \"geometry\"".into()))?;
        let mut geom_map = BTreeMap::new();
        for (k, v) in geometry {
            let n = v
                .as_u64()
                .ok_or_else(|| RepackError::BadManifest(format!("geometry.{k} is not a number")))?;
            geom_map.insert(k.clone(), n);
        }
        let layout = obj
            .get("layout")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RepackError::BadManifest("missing \"layout\"".into()))?
            .iter()
            .map(|e| {
                let o = e
                    .as_object()
                    .ok_or_else(|| RepackError::BadManifest("layout row is not an object".into()))?;
                Ok(LayoutEntry {
                    proj: o
                        .get("proj")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| RepackError::BadManifest("layout row missing proj".into()))?
                        .to_string(),
                    region: o
                        .get("region")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| RepackError::BadManifest("layout row missing region".into()))?
                        .to_string(),
                    off: o
                        .get("off")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| RepackError::BadManifest("layout row missing off".into()))?,
                    len: o
                        .get("len")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| RepackError::BadManifest("layout row missing len".into()))?,
                })
            })
            .collect::<Result<Vec<_>, RepackError>>()?;
        let slices = obj
            .get("slices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RepackError::BadManifest("missing \"slices\"".into()))?
            .iter()
            .map(parse_slice_entry)
            .collect::<Result<Vec<_>, RepackError>>()?;
        let m = Manifest {
            format,
            version: version as u32,
            generated_by,
            geometry: geom_map,
            layout,
            slices,
        };
        m.validate_geometry()?;
        Ok(m)
    }

    /// The manifest's geometry must agree with the compiled-in constants —
    /// a manifest from a different build must not be silently accepted.
    pub fn validate_geometry(&self) -> Result<(), RepackError> {
        // Public structs can be constructed without parse(). The load path must
        // reject v1 even when sizes, offsets and hashes happen to agree.
        if self.format != FORMAT || self.version != VERSION {
            return Err(RepackError::BadManifest(format!(
                "layout version {} / format {:?} is not v{VERSION} / {FORMAT}; repack required",
                self.version, self.format
            )));
        }
        let want: [(&str, u64); 7] = [
            ("hidden", geom::HIDDEN as u64),
            ("intermediate", geom::INTERMEDIATE as u64),
            ("experts_per_layer", geom::EXPERTS_PER_LAYER as u64),
            ("moe_layers", geom::MOE_LAYERS as u64),
            ("ep_ranks", geom::EP_RANKS as u64),
            ("expert_bytes", geom::EXPERT_BYTES as u64),
            ("quarter_slice_bytes", geom::QUARTER_SLICE_BYTES as u64),
        ];
        for (field, w) in want {
            let got = self.geometry.get(field).copied().ok_or_else(|| {
                RepackError::BadManifest(format!("geometry missing {field}"))
            })?;
            if got != w {
                return Err(RepackError::GeometryMismatch {
                    field: leak(field),
                    want: w,
                    got,
                });
            }
        }
        // The layout table must be the pinned one, byte for byte.
        let pinned = geom::layout_table();
        if self.layout.len() != pinned.len() {
            return Err(RepackError::BadManifest(format!(
                "layout has {} rows, expected {}",
                self.layout.len(),
                pinned.len()
            )));
        }
        for (got, want) in self.layout.iter().zip(pinned.iter()) {
            if got.proj != want.proj.name()
                || got.region != want.region
                || got.off != want.off as u64
                || got.len != want.len as u64
            {
                return Err(RepackError::BadManifest(format!(
                    "layout row drift: got {} {} @{} +{}, expected {} {} @{} +{}",
                    got.proj,
                    got.region,
                    got.off,
                    got.len,
                    want.proj.name(),
                    want.region,
                    want.off,
                    want.len
                )));
            }
        }
        Ok(())
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

fn leak(s: &str) -> &'static str {
    match s {
        "hidden" => "hidden",
        "intermediate" => "intermediate",
        "experts_per_layer" => "experts_per_layer",
        "moe_layers" => "moe_layers",
        "ep_ranks" => "ep_ranks",
        "expert_bytes" => "expert_bytes",
        "quarter_slice_bytes" => "quarter_slice_bytes",
        _ => "geometry",
    }
}

fn parse_slice_entry(v: &crate::json::Json) -> Result<SliceEntry, RepackError> {
    let o = v
        .as_object()
        .ok_or_else(|| RepackError::BadManifest("slice entry is not an object".into()))?;
    let get_str = |k: &str| -> Result<String, RepackError> {
        o.get(k)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| RepackError::BadManifest(format!("slice entry missing {k}")))
    };
    let get_u64 = |k: &str| -> Result<u64, RepackError> {
        o.get(k)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RepackError::BadManifest(format!("slice entry missing {k}")))
    };
    let file = get_str("file")?;
    let sha = get_str("sha256")?;
    // A malformed sha must fail at parse time, not compare unequal by luck.
    sha256::unhex(&sha).map_err(|e| RepackError::BadManifest(format!("{file}: bad sha256: {e}")))?;
    let bytes = get_u64("bytes")?;
    let layer = get_u64("layer")? as usize;
    let expert = get_u64("expert")? as usize;
    let rank = get_u64("rank")? as usize;
    let source = o
        .get("source")
        .and_then(|v| v.as_object())
        .ok_or_else(|| RepackError::BadManifest(format!("{file}: missing source")))?;
    let shard = source
        .get("shard")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RepackError::BadManifest(format!("{file}: source missing shard")))?
        .to_string();
    let tensors = source
        .get("tensors")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RepackError::BadManifest(format!("{file}: source missing tensors")))?
        .iter()
        .map(|t| {
            let o = t.as_object().ok_or_else(|| {
                RepackError::BadManifest(format!("{file}: tensor entry is not an object"))
            })?;
            Ok(SourceTensor {
                name: o
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        RepackError::BadManifest(format!("{file}: tensor missing name"))
                    })?
                    .to_string(),
                shape: o
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        RepackError::BadManifest(format!("{file}: tensor missing shape"))
                    })?
                    .iter()
                    .map(|d| d.as_u64().unwrap_or(u64::MAX) as usize)
                    .collect(),
                dtype: o
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        RepackError::BadManifest(format!("{file}: tensor missing dtype"))
                    })?
                    .to_string(),
            })
        })
        .collect::<Result<Vec<_>, RepackError>>()?;
    Ok(SliceEntry {
        file,
        sha256: sha,
        bytes,
        layer,
        expert,
        rank,
        shard,
        tensors,
    })
}

/// Build the six source-tensor references for one expert.
pub fn source_tensors(layer: usize, expert: usize) -> Vec<SourceTensor> {
    let mut out = Vec::with_capacity(6);
    for p in Proj::ALL {
        for scale in [false, true] {
            let shape = if scale {
                vec![p.out_rows(), p.in_cols() / 32]
            } else {
                vec![p.out_rows(), p.in_cols() / 2]
            };
            out.push(SourceTensor {
                name: geom::tensor_name(layer, expert, p, scale),
                shape,
                dtype: "U8".to_string(),
            });
        }
    }
    out
}

/// JSON string escaping (the manifest only ever holds ASCII names, but escape
/// properly so a stray byte cannot produce an unparseable manifest).
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
