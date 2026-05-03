
This repository contains code for processing raw network traffic captures and testing traffic classifiers.

## Dataset

The dataset is not included in this repository. You can download it from the following link:

**Dataset Link:** https://i62nextcloud.tm.kit.edu/index.php/s/QGtA69Riwyw6Sjo

Once you open the link, you will find three folders named after three different obfuscation plugins. Inside each folder, there are obfuscated and non obfuscated PCAP files.

## Installation

Install the required dependencies:

`pip install -r requirements.txt`

## Processing Pipeline

**Step 1:** Run `TracesToFlows.py` to split the PCAP files into individual network flows based on the 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol). 

`python3 TracesToFlows.py`

This script uses SplitCap.exe, download and install SplitCap from the official source.

Output: A directory containing splitted PCAPs (one file per flow).

**Step 2:** Provide the directory with splitted PCAPs to `Table_extended_constructor.py`. This script creates a table containing packet arrival times, packet sizes, packet directions, and SNI (Server Name Indication) for target class labeling:

`python3 Table_extended_constructor.py`

**Step 3:** To extract the TLS payload, run these scripts in order (сonfigure the correct paths before running):

`python3 quic_decryptor.py`

`python3 quic_parser.py`

**Step 4:** After obtaining all tables, run the Jupyter notebook:

`jupyter notebook Scenario_hrftc.ipynb`

## Scripts Overview

- `TracesToFlows.py` - Split raw PCAPs into 5-tuple flows
- `Table_extended_constructor.py` - Build feature table with timing, sizes, directions, and SNI
- `quic_decryptor.py` - Decrypt QUIC traffic
- `quic_parser.py` - Parse TLS payload from QUIC
- `Scenario_hrftc.ipynb` - Classification experiments
