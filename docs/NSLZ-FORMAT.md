# NSLZ - NSL Compression Format

## Overview
NSLZ is NSL's native compression format with self-extracting capability.

## Format Structure

```
┌─────────────────────────────────────┐
│ MAGIC: "NSLZ" (4 bytes)             │
│ VERSION: uint16 (2 bytes)           │
│ FLAGS: uint16 (2 bytes)             │
│   - Bit 0: Self-extracting          │
│   - Bit 1: Encrypted                │
│   - Bit 2: Signed                   │
├─────────────────────────────────────┤
│ FILE COUNT: uint32 (4 bytes)        │
├─────────────────────────────────────┤
│ FILE TABLE:                         │
│   For each file:                    │
│   - Name length: uint16             │
│   - Name: UTF-8 bytes               │
│   - Original size: uint64           │
│   - Compressed size: uint64         │
│   - Offset: uint64                  │
│   - CRC32: uint32                   │
├─────────────────────────────────────┤
│ COMPRESSED DATA BLOCKS              │
│   Block header:                     │
│   - Block type: uint8               │
│     0x00 = Store (no compression)   │
│     0x01 = LZ77 (patterns)          │
│     0x02 = Huffman (entropy)        │
│     0x03 = NSL Semantic             │
│   - Block size: uint32              │
│   - Data: bytes                     │
├─────────────────────────────────────┤
│ FOOTER:                             │
│   - Total CRC32: uint32             │
│   - "ZLSN" (4 bytes) - reverse magic│
└─────────────────────────────────────┘
```

## Self-Extracting Mode

When FLAG bit 0 is set, the file is a Windows executable:

```
┌─────────────────────────────────────┐
│ PE HEADER (Minimal .exe stub)       │
│ - Embedded C# decompressor          │
│ - Size: ~50KB                       │
├─────────────────────────────────────┤
│ NSLZ DATA (as above)                │
└─────────────────────────────────────┘
```

Run the .exe → files extracted to current directory.

## Compression Tiers

### Tier 1: Pattern Recognition (LZ77-based)
- Find repeating byte sequences
- Replace with back-references
- Good for binaries with repeated structures

### Tier 2: Entropy Encoding (Huffman)
- Build frequency table
- Assign shorter codes to common bytes
- Good for text and varied data

### Tier 3: NSL Semantic (Optional)
- Recognize known file signatures (.exe, .dll, .json)
- Apply type-specific compression
- E.g., strip debug symbols, minify JSON

## Usage in NSL

```nsl
// Compress files
nslz.create("output.nslz", ["file1.exe", "file2.dll"])

// Compress as self-extracting
nslz.createSFX("output.exe", ["file1.exe", "file2.dll"])

// Extract
nslz.extract("archive.nslz", "output_dir/")

// List contents
let files = nslz.list("archive.nslz")

// Get info
let info = nslz.info("archive.nslz")
// {files: 5, originalSize: 100MB, compressedSize: 40MB, ratio: 0.4}
```

## Why NSLZ?

1. **Self-extracting**: No tools needed to open
2. **GitHub friendly**: Single file upload
3. **NSL native**: Integrates with NSL ecosystem
4. **Semantic compression**: Smarter than generic ZIP
5. **Extensible**: New compression tiers can be added
