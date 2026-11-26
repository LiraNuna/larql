//! A NumPy `.npy` writer, format version 1.0, little-endian, C order: the file `numpy.load` reads.
//!
//! The header is the magic string, the version, a little-endian u16 header length, then a Python dict literal
//! padded with spaces and ended by a newline so that the data starts on a multiple of [`HEADER_ALIGN`] bytes.
//! The body is streamed: the writer never holds the array. Every byte written is hashed as it goes, so the digest
//! is of exactly the file on disk.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

/// `\x93NUMPY`.
const MAGIC: &[u8] = b"\x93NUMPY";
/// Format version 1.0: a u16 header length.
const VERSION: [u8; 2] = [1, 0];
/// The data starts on a multiple of this many bytes.
const HEADER_ALIGN: usize = 64;
/// Bytes before the dict: magic, version, the u16 length.
const PREAMBLE_BYTES: usize = MAGIC.len() + VERSION.len() + std::mem::size_of::<u16>();
/// Buffered output, so a selection streamed row by row is not a write per row.
const WRITE_BUFFER_BYTES: usize = 1 << 20;

/// The element type of the array, as NumPy's `descr` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDescr {
    /// Little-endian float32.
    F4,
    /// Little-endian float64.
    F8,
    /// Little-endian uint16.
    U2,
    /// Little-endian uint32.
    U4,
}

impl NpyDescr {
    /// NumPy's spelling of the element type: `<f4`, `<f8`, `<u2` or `<u4`.
    pub fn label(self) -> &'static str {
        match self {
            Self::F4 => "<f4",
            Self::F8 => "<f8",
            Self::U2 => "<u2",
            Self::U4 => "<u4",
        }
    }
}

/// The header bytes for an array of `descr` and `shape`.
fn header(descr: NpyDescr, shape: &[usize]) -> std::io::Result<Vec<u8>> {
    let dims = match shape {
        [only] => format!("({only},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {dims}, }}",
        descr.label()
    );
    // Pad with spaces so the newline that ends the dict is the last byte before an aligned data start.
    let unpadded = PREAMBLE_BYTES + dict.len() + 1;
    dict.push_str(&" ".repeat(unpadded.next_multiple_of(HEADER_ALIGN) - unpadded));
    dict.push('\n');
    let length = u16::try_from(dict.len()).map_err(|_| {
        std::io::Error::other(format!(
            "an .npy 1.0 header holds at most {} bytes; this shape needs {}",
            u16::MAX,
            dict.len()
        ))
    })?;
    let mut bytes = Vec::with_capacity(PREAMBLE_BYTES + dict.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    Ok(bytes)
}

/// The SHA-256 of the `.npy` file an array of `descr` and `shape` with this body would be, lowercase hex: the
/// digest [`NpyWriter::finish`] gives for the same array, computed without writing a file.
pub(super) fn npy_sha256(descr: NpyDescr, shape: &[usize], body: &[u8]) -> std::io::Result<String> {
    let mut hash = Sha256::new();
    hash.update(header(descr, shape)?);
    hash.update(body);
    Ok(format!("{:x}", hash.finalize()))
}

/// The whole `.npy` file an array of `descr` and `shape` with this body is, in memory: the bytes
/// [`NpyWriter`] writes for the same array, so their SHA-256 is the `npy_sha256` a typed operation's
/// provenance records. `body` is the elements, little-endian, C order.
pub fn npy_file_bytes(descr: NpyDescr, shape: &[usize], body: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut bytes = header(descr, shape)?;
    bytes.extend_from_slice(body);
    Ok(bytes)
}

/// An `.npy` file being written. Created only where no file exists.
pub(super) struct NpyWriter {
    out: BufWriter<File>,
    hash: Sha256,
}

impl NpyWriter {
    /// Create `path`, which must not exist, and write the header.
    pub(super) fn create_new(
        path: &Path,
        descr: NpyDescr,
        shape: &[usize],
    ) -> std::io::Result<Self> {
        let header = header(descr, shape)?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut writer = Self {
            out: BufWriter::with_capacity(WRITE_BUFFER_BYTES, file),
            hash: Sha256::new(),
        };
        writer.write(&header)?;
        Ok(writer)
    }

    /// Append body bytes: elements already little-endian, of the header's `descr`.
    pub(super) fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.hash.update(bytes);
        self.out.write_all(bytes)
    }

    /// Flush to disk and give the SHA-256 of the whole file, lowercase hex.
    pub(super) fn finish(self) -> std::io::Result<String> {
        let file = self.out.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(format!("{:x}", self.hash.finalize()))
    }
}

/// Decoded float elements from a C-order little-endian `.npy` file.
pub(super) enum NpyFloatData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl NpyFloatData {
    pub(super) fn dtype_label(&self) -> &'static str {
        match self {
            Self::F32(_) => "<f4",
            Self::F64(_) => "<f8",
        }
    }
}

/// Parsed `.npy` float array plus the SHA-256 digest of the file bytes on disk.
pub(super) struct NpyFloats {
    pub(super) shape: Vec<usize>,
    pub(super) data: NpyFloatData,
    pub(super) sha256: String,
}

/// Read a NumPy `.npy` (v1.0/v2.0/v3.0) C-order `<f4` or `<f8` array from `path`.
/// Refuses Fortran-order, unsupported dtypes, malformed headers, or truncated bodies.
pub(super) fn read_npy_floats(path: &Path) -> Result<NpyFloats, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read .npy file {}: {e}", path.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(format!(
            "{} is not a valid .npy file (missing NUMPY magic header)",
            path.display()
        ));
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_start, header_len) = match (major, minor) {
        (1, 0) => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        (2, 0) | (3, 0) => {
            if bytes.len() < 12 {
                return Err(format!("{} has a truncated .npy header", path.display()));
            }
            (
                12usize,
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            )
        }
        _ => {
            return Err(format!(
                "{}: unsupported .npy format version {major}.{minor}",
                path.display()
            ));
        }
    };
    let data_start = header_start
        .checked_add(header_len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| format!("{} has a truncated .npy header", path.display()))?;
    let header_str = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|e| format!("{}: .npy header is not UTF-8: {e}", path.display()))?;

    let descr = extract_npy_string_field(header_str, "descr")
        .ok_or_else(|| format!("{}: .npy header missing 'descr'", path.display()))?;
    let fortran = extract_npy_bool_field(header_str, "fortran_order")
        .ok_or_else(|| format!("{}: .npy header missing 'fortran_order'", path.display()))?;
    if fortran {
        return Err(format!(
            "{}: Fortran-order .npy arrays are not supported (convert with np.ascontiguousarray)",
            path.display()
        ));
    }
    let shape = extract_npy_shape_field(header_str).ok_or_else(|| {
        format!(
            "{}: .npy header has invalid or missing 'shape'",
            path.display()
        )
    })?;

    let is_f64 = match descr.as_str() {
        "<f4" | "f4" => false,
        "<f8" | "f8" => true,
        other => {
            return Err(format!(
                "{}: unsupported .npy dtype `{other}` (expected `<f4` or `<f8`)",
                path.display()
            ));
        }
    };
    let elem_bytes = if is_f64 { 8usize } else { 4usize };
    let total_elems = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("{}: .npy shape {:?} overflows usize", path.display(), shape))?;
    let expected_bytes = total_elems
        .checked_mul(elem_bytes)
        .ok_or_else(|| format!("{}: .npy byte length overflows usize", path.display()))?;
    let payload = &bytes[data_start..];
    if payload.len() != expected_bytes {
        return Err(format!(
            "{}: .npy payload is {} bytes, expected {expected_bytes} for shape {shape:?} dtype `{descr}`",
            path.display(),
            payload.len()
        ));
    }
    let data = if is_f64 {
        NpyFloatData::F64(
            payload
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        )
    } else {
        NpyFloatData::F32(
            payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    };
    Ok(NpyFloats {
        shape,
        data,
        sha256,
    })
}

fn find_npy_key_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let single = format!("'{key}'");
    let double = format!("\"{key}\"");
    let pos = header.find(&single).or_else(|| header.find(&double))?;
    let after_key = &header[pos + key.len() + 2..];
    let colon = after_key.find(':')?;
    Some(after_key[colon + 1..].trim_start())
}

fn extract_npy_string_field(header: &str, key: &str) -> Option<String> {
    let rest = find_npy_key_value(header, key)?;
    let quote = rest.chars().next().filter(|&c| c == '\'' || c == '"')?;
    let inner = &rest[1..];
    let end = inner.find(quote)?;
    Some(inner[..end].to_string())
}

fn extract_npy_bool_field(header: &str, key: &str) -> Option<bool> {
    let rest = find_npy_key_value(header, key)?;
    if rest.starts_with("False") {
        Some(false)
    } else if rest.starts_with("True") {
        Some(true)
    } else {
        None
    }
}

fn extract_npy_shape_field(header: &str) -> Option<Vec<usize>> {
    let rest = find_npy_key_value(header, "shape")?;
    let rest = rest.strip_prefix('(')?;
    let end = rest.find(')')?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        dims.push(p.parse::<usize>().ok()?);
    }
    Some(dims)
}
