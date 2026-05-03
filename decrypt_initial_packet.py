#!/usr/bin/env python3
"""
QUIC Initial decryptor — reads from CSV with multiple packets, uses DCID from first client Initial packet.
Usage:
  python3 decrypt_initial_packet.py --csv packets.csv --out-csv decrypted_quic.csv --pn-kmax 20000 --verbose

The script reads a CSV with 'packet_hex' column, extracts DCID from the first client Initial packet (role=client, packet_number=0),
and attempts to decrypt each packet using that DCID. Successful decryptions are saved to a CSV with columns: direction, packet_num, quic_version, payload.
"""
import sys, argparse, re, binascii, struct
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

# constants
INITIAL_SALTS = {
    0x00000001: bytes.fromhex("38762cf7f55934b34d179ae6a4c80cadccbb7f0a"), # QUIC v1
    0xff00001d: bytes.fromhex("afbfec289993d24c9e9786f19c6111e04390a899"), # draft-29
}
HASHLEN = 32
AEAD_KEYLEN = 16
AEAD_IVLEN = 12
HP_KEYLEN = 16
AES_BLOCK = 16

def hext(b: bytes, n=80):
    if b is None: return "None"
    s = b.hex()
    return s if len(s) <= n else s[:n] + "..."

def read_varint(data: bytes, off: int):
    if off >= len(data): raise IndexError("varint read out of range")
    b0 = data[off]
    prefix = b0 >> 6
    if prefix == 0:
        return b0 & 0x3f, off + 1
    elif prefix == 1:
        if off + 2 > len(data): raise IndexError("varint truncated")
        return ((b0 & 0x3f) << 8) | data[off+1], off + 2
    elif prefix == 2:
        if off + 4 > len(data): raise IndexError("varint truncated")
        return ((b0 & 0x3f) << 24) | (data[off+1] << 16) | (data[off+2] << 8) | data[off+3], off + 4
    else:
        if off + 8 > len(data): raise IndexError("varint truncated")
        v = b0 & 0x3f
        for i in range(1, 8):
            v = (v << 8) | data[off + i]
        return v, off + 8

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if salt is None:
        salt = b'\x00' * HASHLEN
    h = hmac.HMAC(salt, hashes.SHA256())
    h.update(ikm)
    return h.finalize()

def hkdf_expand_label(prk: bytes, label: str, context: bytes, length: int) -> bytes:
    full_label = b"tls13 " + label.encode()
    hkdf_label = struct.pack(">H", length) + bytes([len(full_label)]) + full_label + bytes([len(context)]) + context
    hkexp = HKDFExpand(algorithm=hashes.SHA256(), length=length, info=hkdf_label)
    return hkexp.derive(prk)

def derive_initial_keys_from_ikm(ikm: bytes, version: int, verbose=False, prefix=""):
    salt = INITIAL_SALTS.get(version, INITIAL_SALTS.get(0x00000001))
    if verbose: print(f"{prefix}Using salt for version 0x{version:08x}: {hext(salt)}")
    prk = hkdf_extract(salt, ikm)
    if verbose: print(f"{prefix}PRK = {hext(prk)}")
    
    client_secret = hkdf_expand_label(prk, "client in", b"", HASHLEN)
    server_secret = hkdf_expand_label(prk, "server in", b"", HASHLEN)
    
    client_key = hkdf_expand_label(client_secret, "quic key", b"", AEAD_KEYLEN)
    client_iv  = hkdf_expand_label(client_secret, "quic iv",  b"", AEAD_IVLEN)
    client_hp  = hkdf_expand_label(client_secret, "quic hp",  b"", HP_KEYLEN)
    
    server_key = hkdf_expand_label(server_secret, "quic key", b"", AEAD_KEYLEN)
    server_iv  = hkdf_expand_label(server_secret, "quic iv",  b"", AEAD_IVLEN)
    server_hp  = hkdf_expand_label(server_secret, "quic hp",  b"", HP_KEYLEN)
    
    if verbose:
        print(f"{prefix}client_key={hext(client_key)}, client_iv={hext(client_iv)}, client_hp={hext(client_hp)}")
        print(f"{prefix}server_key={hext(server_key)}, server_iv={hext(server_iv)}, server_hp={hext(server_hp)}")
    
    return {"client": {"key": client_key, "iv": client_iv, "hp": client_hp},
            "server": {"key": server_key, "iv": server_iv, "hp": server_hp}}

def aes_ecb_encrypt_block(key: bytes, block: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    encryptor = cipher.encryptor()
    return encryptor.update(block) + encryptor.finalize()

def try_decrypt(packet: bytes, verbose=True, pn_kmax=4096, client_dcid_override=None):
    if verbose: print("=== START parsing header ===")
    if len(packet) < 1: raise ValueError("packet too short")
    
    first_byte = packet[0]
    is_long = (first_byte & 0x80) != 0
    if not is_long: raise NotImplementedError("short headers are not supported here")
    
    ptype = (first_byte & 0x30) >> 4
    ptype_names = {0: "Initial", 1: "0-RTT", 2: "Handshake", 3: "Retry"}
    if verbose: print(f"first_byte=0x{first_byte:02x}, packet_type={ptype} ({ptype_names.get(ptype,'?')})")
    if ptype != 0 and verbose: print("Warning: packet type is not Initial; continue trying but likely won't decrypt with initial keys")
    
    if len(packet) < 5:
        raise ValueError(f"Packet too short to read version (len={len(packet)})")
    version_num = int.from_bytes(packet[1:5], 'big')
    if verbose: print(f"version: 0x{version_num:08x} ({version_num})")
    pos = 5
    
    if pos >= len(packet):
        raise ValueError(f"Packet too short to read DCID length (pos={pos}, len={len(packet)})")
    dcid_len = packet[pos]
    pos += 1
    if pos + dcid_len > len(packet):
        raise ValueError(f"Packet too short to read DCID (pos={pos}, len={len(packet)})")
    parsed_dcid = packet[pos:pos+dcid_len]
    pos += dcid_len
    
    if pos >= len(packet):
        raise ValueError(f"Packet too short to read SCID length (pos={pos}, len={len(packet)})")
    scid_len = packet[pos]
    pos += 1
    if pos + scid_len > len(packet):
        raise ValueError(f"Packet too short to read SCID (pos={pos}, len={len(packet)})")
    scid = packet[pos:pos+scid_len]
    pos += scid_len
    if verbose: print(f"DCID={parsed_dcid.hex()}, SCID={scid.hex()}")
    
    if pos >= len(packet):
        raise ValueError(f"Packet too short to read token_len varint (pos={pos}, len={len(packet)})")
    try:
        token_len, pos = read_varint(packet, pos)
    except IndexError as e:
        raise ValueError(f"Failed to read token_len varint: {e}")
    
    if pos + token_len > len(packet):
        raise ValueError(f"Packet too short to read token of length {token_len} (pos={pos}, len={len(packet)})")
    token = packet[pos:pos+token_len]
    pos += token_len
    
    if pos >= len(packet):
        raise ValueError(f"Packet too short to read length_val varint (pos={pos}, len={len(packet)})")
    try:
        length_val, pos = read_varint(packet, pos)
    except IndexError as e:
        raise ValueError(f"Failed to read length_val varint: {e}")
    
    pn_offset = pos
    if verbose: print(f"token_len={token_len}, length={length_val}, pn_offset={pn_offset}, packet_len={len(packet)}")
    
    # build a list of ikm candidates for HKDF-Extract
    ikm_candidates = []
    if client_dcid_override:
        ikm_candidates.append(("CLIENT_PROVIDED", client_dcid_override))
        if verbose: print(f"Using provided client-dcid for HKDF input: {client_dcid_override.hex()}")
    else:
        # try server packet's DCID and SCID as fallback
        ikm_candidates.append(("DCID_FROM_PACKET", parsed_dcid))
        ikm_candidates.append(("SCID_FROM_PACKET", scid))
    
    for ikm_name, ikm_val in ikm_candidates:
        if verbose: print(f"\n=== Trying HKDF IKM candidate {ikm_name} ({ikm_val.hex()}) ===")
        keys = derive_initial_keys_from_ikm(ikm_val, version_num, verbose=verbose, prefix=f"  [{ikm_name}] ")
        
        # try both roles (client/server) because we don't know direction in capture
        for role in ("client", "server"):
            k = keys[role]["key"]
            iv = keys[role]["iv"]
            hp = keys[role]["hp"]
            if verbose: print(f"[role={role}] key={hext(k)} iv={hext(iv)} hp={hext(hp)}")
            
            # RFC: sample is taken as if PN field were 4 bytes long
            sample_offset = pn_offset + 4
            if sample_offset + AES_BLOCK > len(packet):
                if verbose: print(f"  not enough bytes for sample at offset {sample_offset}; skipping this role")
                continue
            sample = packet[sample_offset:sample_offset+AES_BLOCK]
            if verbose: print(f"  sample_offset={sample_offset}, sample={hext(sample,64)}")
            
            try:
                mask_block = aes_ecb_encrypt_block(hp, sample)
            except Exception as e:
                if verbose: print(f"  AES-ECB(hp, sample) failed: {e}")
                continue
            
            mask = mask_block[:5]
            if verbose: print(f"  mask={mask.hex()}")
            
            unmasked_first = packet[0] ^ (mask[0] & 0x0f)
            derived_pn_len = (unmasked_first & 0x03) + 1
            if verbose: print(f"  original_first=0x{packet[0]:02x}, unmasked_first=0x{unmasked_first:02x}, derived_pn_len={derived_pn_len}")
            
            # try pn_len candidates
            pn_len_trials = []
            if derived_pn_len not in pn_len_trials: pn_len_trials.append(derived_pn_len)
            for i in range(1,5):
                if i not in pn_len_trials: pn_len_trials.append(i)
            
            for pn_len in pn_len_trials:
                if pn_offset + pn_len > len(packet):
                    if verbose: print(f"    pn_len={pn_len} would overflow, skip")
                    continue
                
                # unmask truncated PN bytes
                try:
                    unmasked_pn = bytes(packet[pn_offset + i] ^ mask[1 + i] for i in range(pn_len))
                except IndexError as e:
                    if verbose: print(f"    mask index error: {e}")
                    continue
                truncated = int.from_bytes(unmasked_pn, 'big')
                if verbose: print(f"    pn_len={pn_len}, truncated={truncated} (hex={unmasked_pn.hex()})")
                
                ciphertext_end = pos + length_val 
                ciphertext = packet[pn_offset + pn_len:ciphertext_end]
                if len(ciphertext) < 16:
                    if verbose: print(f"    ciphertext too short ({len(ciphertext)} bytes) — need at least 16 bytes tag; skip")
                    continue
                
                # offline brute-force for full PN = truncated + k * window
                window = 1 << (pn_len * 8)
                tried = 0
                for k_mult in range(0, pn_kmax):
                    packet_number = truncated + k_mult * window
                    tried += 1
                    
                    # build associated data
                    header_before_pn = bytearray(packet[:pn_offset])
                    header_before_pn[0] = unmasked_first
                    associated_data = bytes(header_before_pn) + unmasked_pn
                    
                    # nonce = iv XOR packet_number (big-endian padded to iv length)
                    try:
                        pn_bytes_for_nonce = packet_number.to_bytes(len(iv), 'big')
                    except OverflowError:
                        continue
                    nonce = bytes(a ^ b for a, b in zip(iv, pn_bytes_for_nonce))
                    
                    if verbose and k_mult < 4:
                        print(f"      try k={k_mult}, packet_number={packet_number}, nonce={hext(nonce)} ciphertext_len={len(ciphertext)}")
                    
                    try:
                        aesgcm = AESGCM(k)
                        plain = aesgcm.decrypt(nonce, ciphertext, associated_data)
                        return {"plaintext": plain, "packet_number": packet_number, "role_used": role, "version": version_num, "ikm_used": ikm_name, "dcid_packet": parsed_dcid.hex(), "scid_packet": scid.hex()}
                    except Exception:
                        pass
                
                if verbose:
                    print(f"    tried {tried} candidates for pn_len={pn_len} (window={window}) without success")
    
    raise ValueError("unable to decrypt packet with derived initial keys (tried ikm candidates, roles, pn candidates)")