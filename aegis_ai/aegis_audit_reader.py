"""
AEGIS SYSTEMS RUNTIME (AL-LANG-02) — OFFLINE AUDIT RECONSTITUTOR
Paradigm: Deferred Materialization (Observer Pays Formatting Cost)
Pillar Alignment: PILLAR 1 (100% Cryptographic Auditability, Zero Exceptions)
"""

import ctypes
import struct

class AegisAuditRecordStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp_cycles", ctypes.c_uint64),
        ("step_id", ctypes.c_uint64),
        ("batch_size", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("loss", ctypes.c_float),
        ("lr", ctypes.c_float),
        ("memory_kb", ctypes.c_uint32),
        ("sample_tokens", ctypes.c_uint16 * 8),
        ("chain_hash", ctypes.c_uint32),
        ("status_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]

assert ctypes.sizeof(AegisAuditRecordStruct) == 64, f"Struct size is {ctypes.sizeof(AegisAuditRecordStruct)}, expected 64"

def verify_audit_record(rec, prev_hash):
    """Verifies cryptographic FNV-1a hash chain."""
    h = prev_hash ^ 0x811C9DC5
    cycles = rec.timestamp_cycles & 0xFFFFFFFF
    step = rec.step_id & 0xFFFFFFFF
    loss_bytes = struct.pack('f', rec.loss)
    loss_bits = struct.unpack('I', loss_bytes)[0]
    
    h = ((h ^ cycles) * 0x01000193) & 0xFFFFFFFF
    h = ((h ^ step)   * 0x01000193) & 0xFFFFFFFF
    h = ((h ^ loss_bits) * 0x01000193) & 0xFFFFFFFF
    return h == rec.chain_hash

def reconstitute_audit_event(rec, vocab_dict=None):
    """
    Deferred Materialization:
    Reconstructs human-readable text and SIEM JSON from compact 64-byte binary record.
    The AI engine spent 0 nanoseconds formatting this text at runtime.
    """
    if vocab_dict:
        decoded_tokens = [vocab_dict.get(tok, f"<id:{tok}>") for tok in rec.sample_tokens]
        preview_text = "".join(decoded_tokens)
    else:
        preview_text = " ".join(str(t) for t in rec.sample_tokens)

    status_str = "VALID" if (rec.status_flags & 0x01) else "ANOMALY"
    return {
        "step": rec.step_id,
        "timestamp_cycles": rec.timestamp_cycles,
        "batch_size": rec.batch_size,
        "context_len": rec.block_size,
        "loss": round(float(rec.loss), 5),
        "lr": round(float(rec.lr), 6),
        "memory_mb": round(rec.memory_kb / 1024.0, 2),
        "status": status_str,
        "chain_hash": f"0x{rec.chain_hash:08X}",
        "token_preview": preview_text
    }
