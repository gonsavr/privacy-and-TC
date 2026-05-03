from typing import Self
import pandas as pd
import numpy as np
import subprocess
from subprocess import check_output
from scapy.all import *


class PacketSequence:
    def __init__(self, pcap_name: str, key_dir: str):
        self.pcap_name = pcap_name
        self.keylog_path = key_dir

        # Save 1st packet for IP 
        self.first_packet = rdpcap(self.pcap_name)[0]

        # Main tshark command
        self._run_main_tshark()

        # Define IP and L4 protocol
        self._detect_ip_and_protocol()

        # tshark command for SNI
        self._extract_sni()

        # TSPS table construction
        self.TSPS = self.TSPS_table_constructor()

    # ==============================
    #  tshark 
    # ==============================
    def _run_main_tshark(self):
        tshark_command = [
            "tshark", "-r", self.pcap_name,
            "-o", f"tls.keylog_file:{self.keylog_path}",
            "-T", "fields",
            "-E", "separator=\t",
            "-e", "frame.number",
            "-e", "frame.time_relative",
            "-e", "ip.src",
            "-e", "ipv6.src",
            "-e", "tcp.len",
            "-e", "udp.length",
            "-e", "quic.frame_type",
            "-e", "tls.record.content_type"
        ]

        output = subprocess.check_output(
            tshark_command,
            universal_newlines=True,
            stderr=subprocess.DEVNULL
        )
        self._raw_lines = output.splitlines()

    # ==============================
    # IP and L4 protocol
    # ==============================
    def _detect_ip_and_protocol(self):
        if 'IP' in self.first_packet:
            self.my_ip = self.first_packet['IP'].src
        else:
            self.my_ip = self.first_packet['IPv6'].src

        has_tcp = False
        has_udp = False
        has_quic = False

        for line in self._raw_lines:
            fields = line.split('\t')
            tcp_len = fields[4]
            udp_len = fields[5]
            quic_ft = fields[6]

            if tcp_len.isdigit() and int(tcp_len) > 0:
                has_tcp = True
            if udp_len.isdigit() and int(udp_len) > 0:
                has_udp = True
            if quic_ft.strip():
                has_quic = True

        if has_tcp:
            self.L4_protocol = 'TCP'
        elif has_quic:
            self.L4_protocol = 'QUIC'
        elif has_udp:
            self.L4_protocol = 'UDP'
        else:
            self.L4_protocol = 'UDP'

    # ==============================
    # SNI only if exists
    # ==============================
    def _extract_sni(self):
        self.SNI = None
        # QUIC SNI
        quic_sni_cmd = [
            "tshark", "-r", self.pcap_name,
            "-Y", "quic.sni",
            "-T", "fields",
            "-e", "quic.sni"
        ]
        try:
            output = check_output(quic_sni_cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if lines:
                self.SNI = lines[0]
                return
        except Exception:
            pass

        # TLS SNI
        tls_sni_cmd = [
            "tshark", "-r", self.pcap_name,
            "-Y", "tls.handshake.extensions_server_name",
            "-T", "fields",
            "-e", "tls.handshake.extensions_server_name"
        ]
        try:
            output = check_output(tls_sni_cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if lines:
                self.SNI = lines[0]
        except Exception:
            pass

    # ==============================
    # TSPS table
    # ==============================
    def TSPS_table_constructor(self):
        times = []
        sizes = []
        is_uplink = []

        for line in self._raw_lines:
            fields = line.split('\t')

            time = fields[1]
            ip_src = fields[2] or fields[3]
            tcp_len = fields[4]
            udp_len = fields[5]

            if self.L4_protocol == 'TCP':
                if not tcp_len.isdigit() or int(tcp_len) == 0:
                    continue
                size = int(tcp_len)
            else:
                if not udp_len.isdigit() or int(udp_len) == 0:
                    continue
                size = int(udp_len) - 8

            times.append(float(time))
            sizes.append(size)
            is_uplink.append(int(ip_src == self.my_ip))

        table = {
            "SNI": self.SNI,
            "L4_protocol": self.L4_protocol,
            "Times": [np.array(times)],
            "Sizes": [np.array(sizes)],
            "Is_uplink": [np.array(is_uplink)],
        }

        return pd.DataFrame(table)

    # ==============================
    # Handshake 
    # ==============================
    def TSPS_handshake(self):
        if self.L4_protocol == 'UDP':
            return len(self.TSPS['Times'][0])

        for line in self._raw_lines:
            fields = line.split('\t')
            frame = fields[0]
            quic_ft = fields[6]
            tls_ct = fields[7]

            if self.L4_protocol == 'TCP' and tls_ct == '23':
                return int(frame)

            if self.L4_protocol == 'QUIC' and quic_ft == '30':
                return int(frame)

        return None
