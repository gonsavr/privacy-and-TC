"""Traces2Flows

This script 
1)gets as an input argument path to a dataset of traces (in a format trace1.pcap, ... , traceN.pcap)
2)splits them into single flows in a fromat (flow1.pcapng, ... , flowM.pcapng) using tool named SplitCap.exe. 
  See https://www.netresec.com/?page=SplitCap
3)removes empty and identical .pcapng files
4)converts .pcapng files to .pcap files for the further processing
"""
import os
from glob import glob
from subprocess import check_output
from tqdm import tqdm
import argparse

"""Reading input arguments: upload_path, save_path, min_flow_size, max_flows_num"""
def add_arguments(parser):

	parser.add_argument(
		"--uppload_path", type=str, 
		help="The path to a dataset of the traces. The supported file format is \".pcap\""
	)
	parser.add_argument(
		"--save_path", type=str, 
		help="The path where the splitted flows are saved"
	)
	parser.add_argument(
		"--min_flow_size", type=int,
		default = "1",
		help="min flow size"
	)
	parser.add_argument(
		"--max_flows_num", type=int,
		default="10000",
		help="max number of flows in .pcap file"
	)

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

MAX_NUM_OF_FLOWS = args.max_flows_num
MIN_SIZE_OF_FLOW = args.min_flow_size
UPPLOAD_PATH = args.uppload_path
SAVE_PATH = args.save_path
if UPPLOAD_PATH[-1] == '/':
	UPPLOAD_PATH+='*.pcap'
PCAP_NAMES = sorted(glob(UPPLOAD_PATH))
print(PCAP_NAMES)

'''
convert pcapng to pcap 
'''

for pcap_name in PCAP_NAMES:
    check_output(["editcap", pcap_name, "-F", "pcap", pcap_name.split('.pcap')[0] + '_tmp.pcap' ])

PCAP_NAMES = sorted(glob(UPPLOAD_PATH.split('.pcap')[0] + '_tmp.pcap'))

"""In this cycle SplitCap.exe tool splits traces into single flows
and saves them as .pcapng files in folders named as initial traces (1st line).
Then, all empty files are removed (2nd line).
"""
#Actually, empty files can be removed beyond the cycle
for pcap_name in PCAP_NAMES: 
	os.system(
		f"""
		mono 0.SplitCap/SplitCap_2-1/SplitCap.exe -p {MAX_NUM_OF_FLOWS} -b {MIN_SIZE_OF_FLOW} -r {pcap_name} -o {SAVE_PATH}{os.path.basename(pcap_name)}-ALL
		find {SAVE_PATH}{os.path.basename(pcap_name)}-ALL -size 0 -print -delete
		"""	
)

"""finddupe.exe tool removes the same .pcap files"""
os.system(f'wine 0.SplitCap/finddupe.exe -del {SAVE_PATH}')

"""SplitCap.exe tool saves the processed flows in a format .pcapng.
However, in some cases tshark and Scapy need them to be saved as .pcap to work correctly. 
So, in this cycle *.pcapng files are converted to *.pcap files.
"""


Variety_of_PCAPs = sorted(glob(f'{SAVE_PATH}*/*.pcap'))
for index, PCAP in tqdm(enumerate(Variety_of_PCAPs)):
    check_output(["editcap", PCAP, "-F", "pcap", PCAP])
