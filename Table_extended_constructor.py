from PacketSequence import PacketSequence
import os
import glob
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

TSPS_hat = ['SNI', 'L4_protocol', 'Times', 'Sizes', 'Is_uplink', 'pcap_name', 'N']


SPLITTED_PCAPS_PATH = ""
KEY_PATH = ""
SAVE_PATH = ""

path = os.path.expanduser(SPLITTED_PCAPS_PATH)
key_folder_path = os.path.expanduser(KEY_PATH)
tsps_csv_filename = SAVE_PATH


def key_path(pcap_path):
    parts = pcap_path.split('/')[-1].split('_')
    base_name = '_'.join(parts[:3])
    return f'{key_folder_path}/{base_name}.txt'


def process_pcap(args):
    pcap_name, key_file = args
    rows = []

    if os.path.getsize(pcap_name) == 0:
        return rows

    try:
        ps = PacketSequence(pcap_name, key_file)

        if ps.SNI is None:
            ps.SNI = "NoSni"

        handshake_packet_num = ps.TSPS_handshake()
        # if handshake_packet_num is None:
        #     return rows

        TSPS = ps.TSPS.copy()
        TSPS['pcap_name'] = pcap_name
        TSPS['N'] = handshake_packet_num
        TSPS['SNI'] = ps.SNI

        for _, row in TSPS.iterrows():
            rows.append(row.tolist())

    except Exception:
        return []

    return rows


def main():
    pcap_files = sorted(glob.glob(f"{path}/*/*.pcap"))
    tasks = [(pcap, key_path(pcap)) for pcap in pcap_files]

    total = len(tasks)
    processed = 0

    with open(tsps_csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(TSPS_hat)
        print("writing hat")
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_pcap, t) for t in tasks]

            for future in as_completed(futures):
                rows = future.result()
                for row in rows:
                    writer.writerow(row)

                processed += 1
                progress = int(processed / total * 100)
                print(
                    f"\r[{'#' * (progress // 2)}{' ' * (50 - progress // 2)}] {progress}%",
                    end='',
                    flush=True
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
