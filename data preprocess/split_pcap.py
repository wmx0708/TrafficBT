import os
from scapy.all import rdpcap
import binascii
import json
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm

# This file is used to extract pcap data for creating traffic graph data. The data format is a dictionary:
# {label: [[(payload/packet_length, time, direction)], [...], [...]]}. Each key is a label, and the value is a list.
# Each element in the list represents data from ten packets in a session. The data for each packet is a tuple
# containing three elements: payload/packet_length, time, and direction.
# The resulting dictionary is finally stored in a JSON file.

# Split pcap data into sessions
def pcapng_to_pcap(pcap_path):
    if pcap_path.endswith("pcapng"):
        cmd2 = f"editcap -F pcap {pcap_path} {pcap_path[:-7]}.pcap"
        os.system(cmd2)
        cmd3 = f"del {pcap_path}"
        os.system(cmd3)
    else:
        pcap_name = pcap_path.split("\\")[-1]
        cmd1 = f"ren {pcap_path} {pcap_name[:-5]}_pcapng.pcap"
        os.system(cmd1)
        cmd2 = f"editcap -F pcap {pcap_path[:-5]}_pcapng.pcap {pcap_path}"
        os.system(cmd2)
        cmd3 = f"del {pcap_path[:-5]}_pcapng.pcap"
        os.system(cmd3)


def split_pcap(pcap_path,directory):
    session_dir = directory + "splitcap\\" + pcap_path.removesuffix(".pcap").removeprefix(directory)
    # Splitting command
    cmd = f"SplitCap.exe -r {pcap_path} -s session -o {session_dir}"
    os.system(cmd)
    print(f"Split processed: {pcap_path}")

def multicpu_split_pcap(directory, cpu_count=61):
    """ Use multiprocessing to process pcap files """
    pcap_files = []

    # Traverse the directory and collect all pcap file paths
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".pcap"):  # Process only .pcap files
                pcap_files.append(os.path.join(dirpath, filename))

    # If an error 'System.IO.InvalidDataException: The stream is not a PCAP file...' occurs,
    # it might be a pcapng file. Use the following function to convert it.
    with Pool(processes=cpu_count) as pool:
        list(tqdm(pool.imap(pcapng_to_pcap, pcap_files), total=len(pcap_files), desc="Converting PCAPNG to PCAP"))
        # pool.map(pcapng_to_pcap, [pcap for pcap in pcap_files])

    # Create a process pool and execute in parallel
    with Pool(processes=cpu_count) as pool:
        list(tqdm(pool.starmap(split_pcap, [(pcap, directory) for pcap in pcap_files]), total=len(pcap_files),
                  desc="Splitting PCAP files"))
if __name__=="__main__":

    # Source data path
    # data_path = "D:\\USTC-TFC2016-master\\Benign\\"
    # data_path = "D:\\USTC-TFC2016-master\\Malware\\"
    # data_path = "D:\\ISCX-VPN-Service\\VPN\\"
    # data_path = "D:\\ISCX-VPN-Service\\NonVPN\\"
    # data_path = "D:\\ISCX-VPN-App\\VPN\\"
    # data_path = "D:\\ISCX-VPN-App\\NonVPN\\"
    # data_path = "D:\\ISCX-VPN-App\\VPN\\"
    # data_path = "D:\\ISCX-VPN-App\\NonVPN\\"
    # data_path = "D:\\ISCX-Tor\\Tor\\"
    # data_path = "D:\\ISCX-Tor\\NonTor\\"
    # data_path = "D:\\CIC_IOT_Dataset2022_Attacks\\Flood\\"
    # data_path = "D:\\CIC_IOT_Dataset2022_Attacks\\RTSP-Brute-Force\\"
    # data_path = "D:\\CIRA-CIC-DoHBrw-2020\\detection\\"
    # data_path = "D:\\CrossPlatform\\android\\"
    # data_path = "D:\\CrossPlatform\\ios\\"
    # data_path = "D:\\NUDT_MobileTraffic\\data1\\"
    # data_path = "D:\\NUDT_MobileTraffic\\data2\\"
    # data_path = "D:\\NUDT_MobileTraffic\\data3\\"
    # data_path = "D:\\NUDT_MobileTraffic\\data4\\"

    datapath_list = [
        "D:\\datacon\\datacon_eta\\train\\",
        "D:\\datacon\\datacon_eta\\test\\",
        "D:\\datacon\\datacon2021_eta\\part1\\sample\\",
        "D:\\datacon\\datacon2021_eta\\part1\\real_data\\",
        "D:\\datacon\\datacon2021_eta\\part2\\train_data\\",
        "D:\\datacon\\datacon2021_eta\\part2\\test_data\\"
    ]
    for data_path in datapath_list:
        split_directory=data_path+"splitcap\\"
        # Split the source data
        multicpu_split_pcap(data_path,cpu_count=61)
