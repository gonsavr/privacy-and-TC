import pandas as pd
import numpy as np
import binascii
from tqdm import tqdm
import traceback

DEBUG = True

def tms(bit):
    return {0:6, 1:14, 2:30, 3:62}[bit]

# ---------------- QUIC CRYPTO FRAME ----------------
class QUICCryptoFrame:
    def __init__(self, payload: bytes):
        self.raw_packet = payload if payload is not None else b""
        self.frame_type = 0
        self.frame_offset_two_most_significant_bits = 0
        self.frame_offset_length = 0
        self.frame_length_two_most_significant_bits = 0
        self.frame_length_length = 0
        self.frame_payload = bytearray()

        if len(self.raw_packet) == 0:
            return

        self.frame_type = int(self.raw_packet[0])
        # offset varint starting at 1
        self.frame_offset_two_most_significant_bits, self.frame_offset_length = self.bit_processing(1)
        L = 1 + ((2 + tms(self.frame_offset_two_most_significant_bits)) // 8)
        # length varint starts at L
        self.frame_length_two_most_significant_bits, self.frame_length_length = self.bit_processing(L)
        L = L + ((2 + tms(self.frame_length_two_most_significant_bits)) // 8)
        # payload bytes follow for frame_length_length bytes
        if L < len(self.raw_packet) and self.frame_length_length > 0:
            end = L + self.frame_length_length
            if end > len(self.raw_packet):
                end = len(self.raw_packet)
            self.frame_payload = self.raw_packet[L:end]
        else:
            self.frame_payload = bytearray()

    def bit_processing(self, L):
        """Decode QUIC varint using the same scheme as original code (safe for truncated)."""
        if L >= len(self.raw_packet):
            return 0, 0
        b = int(self.raw_packet[L])
        first_byte = format(b, '08b')
        two_msb = int(first_byte[0]) * 2 + int(first_byte[1])
        tmp_bits = tms(two_msb)
        tmp_len = (2 + tmp_bits) // 8
        if L + tmp_len > len(self.raw_packet):
            tmp_len = max(0, len(self.raw_packet) - L)
        length_bytes = self.raw_packet[L:L + tmp_len]
        if len(length_bytes) == 0:
            return two_msb, 0
        binary = "".join([f'{int(x):08b}' for x in length_bytes])
        if len(binary) <= 2:
            return two_msb, 0
        binary = binary[2:]
        try:
            val = int(binary, 2)
        except Exception:
            val = 0
        return two_msb, val

# ---------------- collect crypto frames ----------------
def concat_crypto_frames(payload: bytes):
    """Return concatenated CRYPTO frame payloads (only payload data), not whole packet."""
    if payload is None or len(payload) == 0:
        return bytearray()
    pos = 0
    Hello = bytearray()
    while pos < len(payload):
        b = payload[pos]
        if b == 0x06:  # CRYPTO frame type
            try:
                frame = QUICCryptoFrame(payload[pos:])
                if frame.frame_payload:
                    Hello += frame.frame_payload
                # compute step = 1 + offset_varint_len + length_varint_len + frame_payload_len
                step = 1
                step += ((2 + tms(frame.frame_offset_two_most_significant_bits)) // 8)
                step += ((2 + tms(frame.frame_length_two_most_significant_bits)) // 8)
                step += frame.frame_length_length
                pos += max(step, 1)
            except Exception:
                pos += 1
        else:
            pos += 1
    return Hello

# ---------------- small utilities ----------------
def zero_random_utf(payload: bytes):
    if payload is None:
        return np.array([], dtype=np.uint8)
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    if arr.size >= 10:
        arr[6:10] = 0
    return arr

def safe_uint8(a):
    arr = np.asarray(a, dtype=np.int64)
    arr = np.mod(arr, 256)
    return arr.astype(np.uint8)

def aligned_packet_number(packet_number, align_len=4):
    arr = np.array(packet_number, dtype=np.int64)
    if arr.size >= align_len:
        res = arr[-align_len:] % 256
    else:
        pad = np.zeros(align_len - arr.size, dtype=np.int64)
        res = np.concatenate([pad, arr]) % 256
    return res.astype(np.uint8)

# ---------------- CH patch / ECH (kept minimal changes) ----------------
def CH_payload_with_plastered_SNI(ClientHello: np.ndarray) -> np.ndarray:
    if ClientHello is None or len(ClientHello) < 50:
        return ClientHello
    try:
        idx_38 = 38
        sid_index = 39 + int(ClientHello[idx_38])
        if sid_index + 1 >= len(ClientHello):
            return ClientHello
        CS_len = int(ClientHello[sid_index]) * 256 + int(ClientHello[sid_index + 1])
        CS_end = 41 + int(ClientHello[idx_38]) + CS_len
        if CS_end >= len(ClientHello):
            return ClientHello
        CompMet_len = int(ClientHello[CS_end])
        ext_len_start = CS_end + CompMet_len + 1
        if ext_len_start + 1 >= len(ClientHello):
            return ClientHello
        length_of_extensions = int(ClientHello[ext_len_start]) * 256 + int(ClientHello[ext_len_start + 1])
        ext_block = ClientHello[ext_len_start + 2:]
        L = 0
        while L < length_of_extensions and (L + 4) <= len(ext_block):
            ext_type = int(ext_block[L]) * 256 + int(ext_block[L + 1])
            ext_len = int(ext_block[L + 2]) * 256 + int(ext_block[L + 3])
            if ext_type == 0:
                start = ext_len_start + 2 + L + 4
                end = start + ext_len
                if start < len(ClientHello) and end <= len(ClientHello):
                    ClientHello[start:end] = 0
            L += 4 + ext_len
    except Exception:
        if DEBUG:
            traceback.print_exc()
    return ClientHello

def EnryptedCH_payload(ClientHello: np.ndarray, change_len=True,
                       encrypted_extensions=set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281])) -> np.ndarray:
    # minimal safe version: remove encrypted extensions (best-effort) and keep lengths safe
    try:
        if ClientHello is None or len(ClientHello) < 10:
            return ClientHello.astype(np.uint8) if ClientHello is not None else np.array([], dtype=np.uint8)
        idx_38 = 38
        sid_index = 39 + int(ClientHello[idx_38]) if idx_38 < len(ClientHello) else None
        if sid_index is None or sid_index + 1 >= len(ClientHello):
            return ClientHello.astype(np.uint8)
        CS_len = int(ClientHello[sid_index]) * 256 + int(ClientHello[sid_index + 1])
        CS_end = 41 + int(ClientHello[idx_38]) + CS_len
        if CS_end >= len(ClientHello):
            return ClientHello.astype(np.uint8)
        CompMet_len = int(ClientHello[CS_end])
        ext_len_start = CS_end + CompMet_len + 1
        if ext_len_start + 1 >= len(ClientHello):
            return ClientHello.astype(np.uint8)
        length_of_extensions = int(ClientHello[ext_len_start]) * 256 + int(ClientHello[ext_len_start + 1])
        extensions = ClientHello[ext_len_start + 2:]
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return ClientHello.astype(np.uint8)

    new_extensions = np.array([], dtype=np.uint8)
    length_of_delete_extensions = 0
    L = 0
    while L < length_of_extensions:
        if L + 4 > len(extensions):
            break
        ext_type = int(extensions[L]) * 256 + int(extensions[L+1])
        ext_len = int(extensions[L+2]) * 256 + int(extensions[L+3])
        if L + 4 + ext_len > len(extensions):
            break
        if ext_type not in encrypted_extensions:
            new_extensions = np.concatenate([new_extensions, extensions[L:L+4+ext_len]])
        else:
            length_of_delete_extensions += ext_len + 4
        L += 4 + ext_len

    remaining = length_of_extensions - length_of_delete_extensions
    if remaining < 0:
        remaining = 0
    new_length_of_extensions = np.array([(remaining // 256) % 256, remaining % 256], dtype=np.uint8)

    head = ClientHello[:ext_len_start].astype(np.uint8)
    ECH_pay = np.concatenate([head, new_length_of_extensions, new_extensions]).astype(np.uint8)

    if change_len:
        try:
            old_len = int(ClientHello[2]) * 256 + int(ClientHello[3])
            new_len = old_len - length_of_delete_extensions
            new_len = max(0, min(new_len, 65535))
            tmp = ECH_pay.astype(np.uint16)
            if tmp.size > 3:
                tmp[2] = (new_len // 256) % 256
                tmp[3] = new_len % 256
            ECH_pay = tmp.astype(np.uint8)
        except Exception:
            if DEBUG:
                traceback.print_exc()
    return ECH_pay

# ---------------- CH+SH -> fixed vector (310) ----------------
def CH_and_SH_recomp(CH_SH_pair):
    # CH_SH_pair: [ClientHello_array (bytes), ServerHello_array (bytes)]
    # Recompose into fixed-length payload of 310 bytes according to Byte positions.
    ClientHello = CH_SH_pair[0] if CH_SH_pair[0] is not None else np.array([], dtype=np.uint8)
    ServerHello = CH_SH_pair[1] if CH_SH_pair[1] is not None else np.array([], dtype=np.uint8)

    # prefix placeholder [1,0,0,0,0] as in original code to align indices
    try:
        CH = np.concatenate([np.array([1,0,0,0,0], dtype=np.uint8), np.array(ClientHello, dtype=np.uint8)])
    except Exception:
        CH = np.array(ClientHello, dtype=np.uint8)
    try:
        SH = np.concatenate([np.array([0,0,0,0,0], dtype=np.uint8), np.array(ServerHello, dtype=np.uint8)])
    except Exception:
        SH = np.array(ServerHello, dtype=np.uint8)

    payload = np.zeros(310, dtype=np.uint8)

    # Dictionaries from original mapping
    CH_L_dict = {0: 123, 16: 125, 21: 127, 35: 129, 41: 131, 51: 133, 44: 135, 25:137}
    CH_D_dict = {
        3: {'position': 139, 'len':2},  15: {'position': 141, 'len':2}, 45: {'position': 143, 'len':2},
        27: {'position': 145, 'len':4},  28: {'position': 149, 'len':4},  6: {'position': 153, 'len':4},
        11: {'position': 157, 'len':4},  19: {'position': 161, 'len':4}, 20: {'position': 165, 'len':4},
        58: {'position': 169, 'len':4},  43: {'position': 173, 'len':12}, 16: {'position':306, 'len':4},
        10: {'position': 185, 'len':26}, 13: {'position': 211, 'len':26}
    }

    # Fill CH part (best-effort with bounds checks)
    try:
        if CH.size >= 5:
            # RV (bytes 0:4 ← CH[1:5])
            if CH.size >= 5:
                payload[0:4] = CH[1:5] if CH.size >= 5 else 0
            # ML, MV (payload[4:8] ← CH[7:11]) 
            if CH.size >= 11:
                payload[4:8] = CH[7:11]
            # SID len
            if CH.size > 43:
                payload[8] = CH[43]
            # cipher suites length: compute as half of (bytes at 44+SID,45+SID)
            sid = int(CH[43]) if CH.size > 43 else 0
            idx_ciph_len = 44 + sid
            if CH.size > idx_ciph_len + 1:
                try:
                    c_len = int(CH[idx_ciph_len]) * 256 + int(CH[idx_ciph_len + 1])
                except Exception:
                    c_len = 0
                payload[9] = (c_len // 2) if c_len > 0 else 0
            # copy some cipher suites area (best-effort)
            L = 46 + sid
            if L < CH.size:
                # r = payload[11]*2 if payload[11] <= 35 else 70
                # but payload[11] initially zero; we'll try to copy safely: attempt copy up to 70 bytes
                r = 0
                if payload[11] != 0:
                    r = int(payload[11]) * 2
                else:
                    r = min(70, max(0, CH.size - L))
                end = L + r
                if end <= CH.size:
                    payload[10:10 + r] = CH[L:L + r]
                # attempt to advance L by sizes used in original logic (best-effort)
            # Extensions len copy
            if L + 2 <= CH.size:
                payload[80:82] = CH[L:L+2]
                L += 2
            # now iterate extensions and map
            type_position = 83
            end_of_CH = int(payload[80]) * 256 + int(payload[81]) + L if (payload[80] != 0 or payload[81] != 0) else L
            while L < CH.size and L < end_of_CH:
                if L + 4 > CH.size:
                    break
                ext_type = int(CH[L]) * 256 + int(CH[L+1])
                ext_len = int(CH[L+2]) * 256 + int(CH[L+3])
                # copy type at type_position
                if type_position + 2 <= 123:
                    if (L + 2) <= CH.size:
                        payload[type_position:type_position+2] = CH[L:L+2]
                    type_position += 2 if type_position < 123 else 0
                if ext_type in CH_L_dict:
                    pos = CH_L_dict[ext_type]
                    if pos + 2 <= payload.size and L+2+1 < CH.size:
                        payload[pos:pos+2] = CH[L+2:L+4]
                    L += 4 + ext_len
                    continue
                elif ext_type in CH_D_dict:
                    pos = CH_D_dict[ext_type]['position']
                    R = min(ext_len, CH_D_dict[ext_type]['len'])
                    if pos + R <= payload.size and L+4+R <= CH.size:
                        payload[pos:pos+R] = CH[L+4:L+4+R]
                    L += 4 + ext_len
                    continue
                else:
                    L += 4 + ext_len
            payload[82] = (type_position - 83)//2 if type_position > 83 else 0
    except Exception:
        if DEBUG:
            traceback.print_exc()

    # Fill SH part (best-effort)
    try:
        if SH.size > 49:
            # Record/version etc
            if SH.size >= 11:
                payload[237:241] = SH[1:5] if SH.size >= 5 else 0
                payload[241:245] = SH[7:11] if SH.size >= 11 else 0
            if SH.size > 43:
                payload[245] = SH[43]
            # cipher
            sid_sh = int(SH[43]) if SH.size > 43 else 0
            idx_ciph_sh = 44 + sid_sh
            if SH.size > idx_ciph_sh + 1:
                payload[246:248] = SH[idx_ciph_sh:idx_ciph_sh+2]
            # ext len
            idx_ext_len = 47 + sid_sh
            if SH.size > idx_ext_len + 1:
                payload[248:250] = SH[idx_ext_len:idx_ext_len+2]
            # iterate SH extensions mapping analogous to CH
            SH_L_dict = {41: 294, 51: 296}
            SH_D_dict = {43: {'position': 299 , 'len':2}, 51: {'position': 301, 'len':2}}
            L = 49 + sid_sh
            end_of_SH = int(payload[248]) * 256 + int(payload[249]) + L if (payload[248] != 0 or payload[249] != 0) else L
            type_position = 250
            while L < SH.size and L < end_of_SH:
                if L + 4 > SH.size:
                    break
                ext_type = int(SH[L]) * 256 + int(SH[L+1])
                ext_len = int(SH[L+2]) * 256 + int(SH[L+3])
                # write type
                if type_position + 2 <= 290:
                    payload[type_position:type_position+2] = SH[L:L+2]
                    type_position += 2 if type_position < 290 else 0
                if ext_type in SH_L_dict:
                    pos = SH_L_dict[ext_type]
                    if pos + 2 <= payload.size and L+2+1 < SH.size:
                        payload[pos:pos+2] = SH[L+2:L+4]
                    L += 4 + ext_len
                    continue
                elif ext_type in SH_D_dict:
                    pos = SH_D_dict[ext_type]['position']
                    R = min(ext_len, SH_D_dict[ext_type]['len'])
                    if pos + R <= payload.size and L+4+R <= SH.size:
                        payload[pos:pos+R] = SH[L+4:L+4+R]
                    L += 4 + ext_len
                    continue
                else:
                    L += 4 + ext_len
            if type_position > 251:
                payload[290] = (type_position - 251)//2 if type_position > 251 else 0
    except Exception:
        if DEBUG:
            traceback.print_exc()

    return payload.astype(np.uint8)

# ---------------- Main processing per pcap_name ----------------
def process_quic_dataset_final(csv_path: str, label_column="SNI", pcap_name_column="pcap_name"):
    df = pd.read_csv(csv_path)
    # fill SNI if needed
    def extract_sni_from_name(x):
        try:
            return x.split("/")[7].split("_")[0]
        except Exception:
            return ""
    if 'SNI' not in df.columns:
        df['SNI'] = df[pcap_name_column].apply(lambda x: extract_sni_from_name(str(x)))
    grouped = df.groupby(pcap_name_column)
    Data, Labels, Pcaps = [], [], []

    for pcap_name, flow_df in tqdm(grouped, desc="Processing flows"):
        flow_df = flow_df.sort_values("packet_num").reset_index(drop=True)
        try:
            # Client
            client_crypto = bytearray()
            client_rows = flow_df[flow_df["direction"] == "client"]
            for _, row in client_rows.iterrows():
                raw = str(row["payload"]).replace("0x","").strip()
                if not raw:
                    continue
                try:
                    payload_bytes = binascii.unhexlify(raw)
                except Exception:
                    continue
                frame_payload = concat_crypto_frames(payload_bytes)
                if frame_payload:
                    client_crypto += frame_payload

            # Server
            server_crypto = bytearray()
            server_rows = flow_df[flow_df["direction"] == "server"]
            for _, row in server_rows.iterrows():
                raw = str(row["payload"]).replace("0x","").strip()
                if not raw:
                    continue
                try:
                    payload_bytes = binascii.unhexlify(raw)
                except Exception:
                    continue
                frame_payload = concat_crypto_frames(payload_bytes)
                if frame_payload:
                    server_crypto += frame_payload

            if len(client_crypto) == 0 or len(server_crypto) == 0:
                if DEBUG:
                    print(f"Flow {pcap_name}: skipped (no crypto on one side). client_len={len(client_crypto)}, server_len={len(server_crypto)}")
                continue

            # Zero random bytes and patch/chop
            client_arr = zero_random_utf(bytes(client_crypto))
            client_arr = CH_payload_with_plastered_SNI(client_arr)
            client_arr = EnryptedCH_payload(client_arr)

            server_arr = zero_random_utf(bytes(server_crypto))

            # Recompose fixed payload
            recomposed = CH_and_SH_recomp([client_arr, server_arr])
            if recomposed is None or recomposed.size == 0:
                if DEBUG:
                    print(f"Flow {pcap_name}: recomposition returned empty")
                continue

            Data.append(recomposed)
            Labels.append(flow_df.iloc[0].get(label_column, ""))
            Pcaps.append(pcap_name)

            if DEBUG:
                print(f"Flow {pcap_name}: OK client_len={len(client_arr)} server_len={len(server_arr)} recomposed_len={len(recomposed)}")

        except Exception:
            print(f"Flow {pcap_name}: unexpected processing error")
            if DEBUG:
                traceback.print_exc()
            continue

    return np.array(Data, dtype=object), np.array(Labels, dtype=object), np.array(Pcaps, dtype=object)

# ---------------- Save ----------------
def save_results_csv(data_array, labels, pcap_names, out_csv="parsed_quic_recomposed.csv"):
    # Save Flow as decimal bytes separated by space
    rows = []
    for row in data_array:
        if isinstance(row, np.ndarray):
            rows.append(' '.join(map(str, row.tolist())))
        else:
            rows.append('')
    df = pd.DataFrame({'Flow': rows, 'labels': labels, 'pcap_name': pcap_names})
    df.to_csv(out_csv, index=False)
    print("Saved to", out_csv)

# ---------------- run ----------------
if __name__=="__main__":
    csv_path = "decrypted_payload.csv"
    OUTPUT_PATH = ""
    Data, Labels, Pcaps = process_quic_dataset_final(csv_path, label_column="SNI", pcap_name_column="pcap_name")
    if len(Data)>0:
        save_results_csv(Data, Labels, Pcaps, out_csv=OUTPUT_PATH)
    else:
        print("No processed flows")
