import os
from scapy.all import rdpcap
import binascii
import json
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm

#本文件用于提取创建流量图数据的pcap数据，数据格式为字典，{label:[[(payload\packet_length,time,direction)],[...],[...]]}每个键是标签，值是一个列表，每个列表的元素是一个session中的十个数据包的数据，每个数据包的数据是一个元组，元组包含payload\\packet_length,time,direction共三个元素，最终将所得字典存储于json文件中
#对pcap数据分流

def pcapng_to_pcap(pcap_path):
    if pcap_path.endswith("pcapng"):
        cmd2 = f"editcap -F pcap {pcap_path} {pcap_path[:-7]}.pcap"
        os.system(cmd2)
        cmd3 = f"del {pcap_path}"
        os.system(cmd3)
    else:
        pcap_name = pcap_path.split("\\")[-1]
        cmd1 = f"ren {pcap_path} {pcap_name[:-5]}_pcapng.pcap"
        os.system(cmd1)  # 执行分流命令
        cmd2 = f"editcap -F pcap {pcap_path[:-5]}_pcapng.pcap {pcap_path}"
        os.system(cmd2)
        cmd3 = f"del {pcap_path[:-5]}_pcapng.pcap"
        os.system(cmd3)


def split_pcap(pcap_path,directory):
    session_dir = directory + "splitcap\\" + pcap_path.removesuffix(".pcap").removeprefix(directory)
    # 分流命令
    cmd = f"SplitCap.exe -r {pcap_path} -s session -o {session_dir}"
    os.system(cmd)
    print(f"Split processed: {pcap_path}")

def multicpu_split_pcap(directory, cpu_count=61):
    """ 使用多进程处理 pcap 文件 """
    pcap_files = []

    # 遍历目录，收集所有 pcap 文件路径
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".pcap"):  # 仅处理 .pcap 文件
                pcap_files.append(os.path.join(dirpath, filename))

    # 创建进程池并并行执行
    #如果分流出现报错‘System.IO.InvalidDataException: The stream is not a PCAP file. Magic number is A0D0D0A or A0D0D0A but should be A1B2C3D4.’可以使用file xxx.pcap查看pcap文件类型是否为pcapng格式，若是，使用下列函数
    with Pool(processes=cpu_count) as pool:
        list(tqdm(pool.imap(pcapng_to_pcap, pcap_files), total=len(pcap_files), desc="Converting PCAPNG to PCAP"))
        # pool.map(pcapng_to_pcap, [pcap for pcap in pcap_files])

    # 创建进程池并并行执行
    with Pool(processes=cpu_count) as pool:
        list(tqdm(pool.starmap(split_pcap, [(pcap, directory) for pcap in pcap_files]), total=len(pcap_files),
                  desc="Splitting PCAP files"))
if __name__=="__main__":

    # 源数据路径
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
        #将源数据分流
        multicpu_split_pcap(data_path,cpu_count=61)


