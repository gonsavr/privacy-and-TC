import pandas as pd
from scapy.all import rdpcap, UDP

packet_num = 7

def is_quic(payload: bytes) -> bool:
    if len(payload) < 5:
        return False
    first = payload[0]
    if (first & 0x80) == 0:
        return False
    version = int.from_bytes(payload[1:5], 'big')
    if version == 0:
        return False
    return True

def process_pcap(pcap_file: str, csv_file: str = 'packets.csv') -> None:
    packets = rdpcap(pcap_file)
    quic_payloads = []
    
    for pkt in packets:
        if UDP in pkt:
            payload = bytes(pkt[UDP].payload)
            if is_quic(payload):
                quic_payloads.append(payload)
                if len(quic_payloads) == packet_num:
                    break
    
    df = pd.DataFrame({'packet_hex': [p.hex() for p in quic_payloads]})
    # df.to_csv(csv_file, index=False)
    # print(f"Wrote {len(quic_payloads)} QUIC packets to {csv_file}")
    
    return df