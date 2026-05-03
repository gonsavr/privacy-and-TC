from decrypt_initial_packet import try_decrypt
from pcap_to_csv import process_pcap
import glob
import pandas as pd
import re, binascii
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os

OUTPUT = "decrypted.csv"
SPLITTED_PCAP_PATH = ""
HEADER = [
    "direction",
    "packet_num",
    "quic_version",
    "payload",
    "pcap_name"
]

def process_single_pcap(pcap_file):
    rows = []
    try:
        processed_pcap = process_pcap(pcap_file)
        if processed_pcap.empty:
            return rows
        
        client_dcid_override = None
        verbose = 0
        pn_kmax = 4096
        
        for i, row in processed_pcap.iterrows():
            try:
                packet_hex = re.sub(r'[^0-9a-fA-F]', '', str(row['packet_hex']))
                packet = binascii.unhexlify(packet_hex)
                if len(packet) < 6:
                    continue
            except Exception:
                continue
            
            try:
                res = try_decrypt(
                    packet,
                    verbose=verbose,
                    pn_kmax=pn_kmax,
                    client_dcid_override=client_dcid_override
                )
                if (
                    res["role_used"] == 'client'
                    and res["packet_number"] == 0
                    and client_dcid_override is None
                ):
                    client_dcid_override = binascii.unhexlify(res["dcid_packet"])
                
                rows.append([
                    res['role_used'],
                    res['packet_number'],
                    f"0x{res['version']:08x}",
                    res['plaintext'].hex(),
                    pcap_file
                ])
            except ValueError:
                continue
    except Exception:
        return []
    
    return rows

def main():
    files = sorted(
        glob.glob(SPLITTED_PCAP_PATH)
    )
    total = len(files)
    processed = 0
    
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_single_pcap, f) for f in files]
            
            for future in as_completed(futures):
                rows = future.result()
                for row in rows:
                    writer.writerow(row)
                
                processed += 1
                progress = int(processed / total * 100)
                print(
                    f"\r[{'#'*(progress//2)}{' '*(50-progress//2)}] {progress}%",
                    end="",
                    flush=True
                )
    
    print("\nDone.")

if __name__ == "__main__":
    main()