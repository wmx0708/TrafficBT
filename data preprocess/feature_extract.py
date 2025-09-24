#  1. 导入依赖模块
import binascii
from scapy.all import rdpcap, Raw
import multiprocessing
from collections import deque
from scapy.layers.inet import TCP, UDP, IP
from collections import defaultdict
import statistics
import math
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ==== 流统计特征 ====
def calculate_iat(timestamps):
    """计算时间间隔（IAT）的统计特征"""
    if len(timestamps) < 2:
        return (0, 0, 0, 0)  # 返回长度为 4 的元组
    iats = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    return (
        sum(iats) / len(iats),  # Mean
        statistics.stdev(iats) if len(iats) > 1 else 0,  # Std
        max(iats),  # Max
        min(iats),  # Min
        sum(iats)  # Total
    )


def calculate_packet_stats(lengths):
    """计算包长度的统计特征"""
    if not lengths:
        return (0, 0, 0, 0, 0)
    return (
        min(lengths),  # Min
        max(lengths),  # Max
        sum(lengths) / len(lengths),  # Mean
        statistics.stdev(lengths) if len(lengths) > 1 else 0,  # Std
        statistics.variance(lengths) if len(lengths) > 1 else 0  # Variance
    )

# ==== 包特征特征计算函数部分 ====
def calculate_entropy(data):
    if not data:
        return 0
    data = bytearray(data)
    probs = [float(data.count(b)) / len(data) for b in set(data)]
    return -sum([p * math.log(p, 2) for p in probs])

def chi_square(data):
    if not data:
        return 0
    expected = len(data) / 256.0
    freq = [0]*256
    for b in data:
        freq[b] += 1
    return sum((f - expected) ** 2 / expected for f in freq)

def printable_ratio(data):
    if not data:
        return 0
    return sum(32 <= b <= 126 for b in data) / len(data)

def null_byte_ratio(data):
    if not data:
        return 0
    return data.count(0) / len(data)

def byte_pair_corr(data):
    if len(data) < 2:
        return 0
    # Ensure the data is numeric and not just raw byte stream
    try:
        a = np.array(data[:-1], dtype=np.float32)
        b = np.array(data[1:], dtype=np.float32)
    except ValueError:
        return 0  # If conversion fails, return 0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0
    return np.corrcoef(a, b)[0, 1]

# ==== TLS解析辅助函数（简化版） ====
def parse_tls(packet):
    tls_info = {"tls_record_type": -1, "tls_version": -1, "cipher_suite_len": -1, "handshake_phase": 0, "is_handshake": False}
    if not packet.haslayer(Raw):
        return tls_info
    data = bytes(packet[Raw].load)
    if len(data) < 5:
        return tls_info
    content_type = data[0]
    version = int.from_bytes(data[1:3], "big")
    tls_info["tls_record_type"] = content_type
    tls_info["tls_version"] = version
    if content_type == 22:  # handshake
        tls_info["is_handshake"] = True
        if len(data) >= 6:
            handshake_type = data[5]
            if handshake_type == 1:
                tls_info["handshake_phase"] = 1  # ClientHello
            elif handshake_type == 2:
                tls_info["handshake_phase"] = 2  # ServerHello
            elif handshake_type == 24:
                tls_info["handshake_phase"] = 3  # KeyUpdate (TLS1.3)
    return tls_info


# ==== 时间特征标准化函数 ====
def normalize_timestamp(timestamps):
    """时间戳标准化（修复numpy与Decimal冲突问题）"""
    # 转换为原生Python float类型
    timestamps_float = [float(ts) for ts in timestamps]

    # 计算时间范围
    min_ts = min(timestamps_float)
    max_ts = max(timestamps_float)
    duration = max_ts - min_ts if max_ts > min_ts else 1.0  # 防止除零

    # 归一化处理
    return [(ts - min_ts) / duration for ts in timestamps_float]

 # === payload 转化成bigram形式的payload ===
def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%4
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def bigram_generation(packet_string,flag=False):
    result = ''
    sentence = cut(packet_string,1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > 256:
                break
            else:
                merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    if flag == True:
        result = result.rstrip()

    return result

# === 流统计特征提取函数 ===
def extract_all_features(packets):
    """主函数：从PCAP文件中提取特征"""
    if not packets:
        return {"stat_features": None}

    # 初始化统计变量
    tcp_flags = defaultdict(int)
    fwd_packets = []
    bwd_packets = []
    all_packets = []
    flag_masks = {
        'F': 0x01, 'S': 0x02, 'R': 0x04, 'P': 0x08,
        'A': 0x10, 'U': 0x20, 'C': 0x80, 'E': 0x40
    }

    # 确定流方向（以第一个包为基准）
    first_pkt = packets[0]
    src_ip = first_pkt[IP].src
    dst_ip = first_pkt[IP].dst
    sport = first_pkt[TCP].sport if TCP in first_pkt else first_pkt[UDP].sport
    dport = first_pkt[TCP].dport if TCP in first_pkt else first_pkt[UDP].dport

    # 处理所有包
    for pkt in packets:
        if IP not in pkt:
            continue

        # 记录TCP标志位
        if TCP in pkt:
            flags = pkt[TCP].flags
            for flag, mask in flag_masks.items():
                if flags & mask:
                    tcp_flags[flag] += 1

        # 确定包方向并记录
        pkt_length = len(pkt)
        timestamp = float(pkt.time)
        if (pkt[IP].src, sport) == (src_ip, sport):  # 前向包
            fwd_packets.append((timestamp, pkt_length))
        else:  # 后向包
            bwd_packets.append((timestamp, pkt_length))
        all_packets.append((timestamp, pkt_length))

    # 按时间排序
    all_packets.sort(key=lambda x: x[0])
    if not all_packets:
        return {"stat_features": None}

    # 计算基础特征
    start_time = all_packets[0][0]
    end_time = all_packets[-1][0]
    flow_duration = (end_time - start_time) * 1e6  # 微秒

    # 包长度统计
    fwd_lengths = [p[1] for p in fwd_packets]
    bwd_lengths = [p[1] for p in bwd_packets]
    all_lengths = [p[1] for p in all_packets]

    # 时间间隔统计
    fwd_times = [p[0] for p in fwd_packets]
    bwd_times = [p[0] for p in bwd_packets]
    all_times = [p[0] for p in all_packets]

    # 计算统计量（假设这些函数已定义）
    fwd_stats = calculate_packet_stats(fwd_lengths)
    bwd_stats = calculate_packet_stats(bwd_lengths)
    all_stats = calculate_packet_stats(all_lengths)
    fwd_iat = calculate_iat(fwd_times)
    bwd_iat = calculate_iat(bwd_times)
    all_iat = calculate_iat(all_times)


    # 构建特征字典
    features = {}

    # 基础信息
    features['Flow Duration'] = flow_duration
    features['total Fwd Packet'] = len(fwd_packets)
    features['total Bwd packets'] = len(bwd_packets)
    features['total Length of Fwd Packet'] = sum(fwd_lengths)
    features['total Length of Bwd Packet'] = sum(bwd_lengths)

    # 前向包统计
    features['Fwd Packet Length Min'] = fwd_stats[0]
    features['Fwd Packet Length Max'] = fwd_stats[1]
    features['Fwd Packet Length Mean'] = fwd_stats[2]
    features['Fwd Packet Length Std'] = fwd_stats[3]

    # 后向包统计
    features['Bwd Packet Length Min'] = bwd_stats[0]
    features['Bwd Packet Length Max'] = bwd_stats[1]
    features['Bwd Packet Length Mean'] = bwd_stats[2]
    features['Bwd Packet Length Std'] = bwd_stats[3]

    # 流量速率
    features['Flow Bytes/s'] = (sum(all_lengths) / flow_duration) * 1e6 if flow_duration > 0 else 0
    features['Flow Packets/s'] = (len(all_packets) / flow_duration) * 1e6 if flow_duration > 0 else 0

    # IAT统计
    features['Flow IAT Mean'] = all_iat[0]
    features['Flow IAT Std'] = all_iat[1]
    features['Flow IAT Max'] = all_iat[2]
    features['Flow IAT Min'] = all_iat[3]

    # 前向IAT
    features['Fwd IAT Mean']= fwd_iat[0]
    features['Fwd IAT Std']= fwd_iat[1] if len(fwd_iat) > 1 else 0
    features['Fwd IAT Max']= fwd_iat[2] if len(fwd_iat) > 2 else 0
    features['Fwd IAT Min']= fwd_iat[3] if len(fwd_iat) > 3 else 0
    features['Fwd IAT Total']= fwd_iat[4] if len(fwd_iat) > 4 else 0

    # 后向IAT
    features['Bwd IAT Mean']= bwd_iat[0]
    features['Bwd IAT Std']= bwd_iat[1] if len(bwd_iat) > 1 else 0
    features['Bwd IAT Max']= bwd_iat[2] if len(bwd_iat) > 2 else 0
    features['Bwd IAT Min']= bwd_iat[3] if len(bwd_iat) > 3 else 0
    features['Bwd IAT Total']= bwd_iat[4] if len(bwd_iat) > 4 else 0

    # TCP标志位
    features['FIN Flag Count'] = tcp_flags.get('F', 0)
    features['SYN Flag Count'] = tcp_flags.get('S', 0)
    features['RST Flag Count'] = tcp_flags.get('R', 0)
    features['PSH Flag Count'] = tcp_flags.get('P', 0)
    features['ACK Flag Count'] = tcp_flags.get('A', 0)
    features['URG Flag Count'] = tcp_flags.get('U', 0)
    features['CWR Flag Count'] = tcp_flags.get('C', 0)
    features['ECE Flag Count'] = tcp_flags.get('E', 0)

    # 全局包统计
    features['Packet Length Min'] = all_stats[0]
    features['Packet Length Max'] = all_stats[1]
    features['Packet Length Mean'] = all_stats[2]
    features['Packet Length Std'] = all_stats[3]
    features['Packet Length Variance'] = all_stats[4]

    return {
        "stat_features": features
    }


# ==== 包级特征提取函数 ====
def extract_packet_features(packets):
    features = []
    payloads = []  # 新增：存储前5个包的payload
    pkt_lens, deltas, directions = deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)

    # 统一使用float类型处理时间戳
    start_time = float(packets[0].time)
    prev_time = start_time
    prev_seq = None
    last_handshake_time = None
    client_ip = packets[0][IP].src if IP in packets[0] else None
    key_update_count = 0

    # 获取时间戳并标准化
    timestamps = [float(pkt.time) for pkt in packets]
    normalized_timestamps = normalize_timestamp(timestamps)

    # 新增：提取前5个包的payload
    for i in range(5):  # 确保只处理前5个包
        pkt = packets[i]
        if bytes(pkt.payload):
            payload = binascii.hexlify(bytes(pkt.payload)).decode('utf-8')
            bigram_payload = bigram_generation(payload)
        payloads.append(bigram_payload)

    # 处理所有包的特征（保持原逻辑）
    for i, pkt in enumerate(packets):
        feat = {}
        if not pkt.haslayer(IP):
            continue

    # 统一使用float类型处理时间戳
    start_time = float(packets[0].time)  # 转换为Python float
    prev_time = start_time
    prev_seq = None
    last_handshake_time = None
    client_ip = packets[0][IP].src if IP in packets[0] else None
    key_update_count = 0

    # 获取时间戳并标准化
    timestamps = [float(pkt.time) for pkt in packets]  # 显式转换为float
    normalized_timestamps = normalize_timestamp(timestamps)

    for i, pkt in enumerate(packets):
        feat = {}
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]

        # ==== 时间特征 ====
        current_time = float(pkt.time)  # 确保为Python float
        feat["timestamp"] = normalized_timestamps[i]
        feat["delta_time"] = current_time - prev_time
        feat["relative_time"] = current_time - start_time  # 直接使用float运算
        prev_time = current_time

        # ==== 长度与方向特征 ====
        plen = len(pkt)
        feat["packet_length"] = int(plen)  # 转换为Python int

        payload_len = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0
        feat["payload_length"] = int(payload_len)

        direction = 1 if ip.src == client_ip else -1
        feat["direction"] = int(direction)
        feat["is_ack_only"] = int(pkt.haslayer(TCP) and pkt[TCP].flags & 0x10 and payload_len == 0)

        # ==== 协议特征 ====
        feat["protocol_id"] = int(ip.proto)  # 转换为Python int

        # ==== TCP特征 ====
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            flags = int(tcp.flags)  # 显式转换为int
            feat["tcp_flag_syn"] = int((flags & 0x02) != 0)
            feat["tcp_flag_ack"] = int((flags & 0x10) != 0)
            feat["tcp_flag_fin"] = int((flags & 0x01) != 0)

            # 处理序列号差值
            current_seq = int(tcp.seq)
            if prev_seq is not None:
                feat["seq_diff"] = int(current_seq - prev_seq)
            else:
                feat["seq_diff"] = 0
            prev_seq = current_seq

            feat["window_size"] = int(tcp.window)
        else:
            feat.update({
                "tcp_flag_syn": 0,
                "tcp_flag_ack": 0,
                "tcp_flag_fin": 0,
                "seq_diff": 0,
                "window_size": 0
            })

        # ==== 滑动窗口统计 ====
        pkt_lens.append(plen)
        deltas.append(feat["delta_time"])
        directions.append(direction)

        # 转换为Python原生类型
        feat["avg_pkt_len_last_5"] = float(np.mean(pkt_lens))
        feat["avg_delta_time_last_5"] = float(np.mean(deltas))
        feat["std_pkt_len_last_5"] = float(np.std(pkt_lens)) if len(pkt_lens) >= 2 else 0.0
        feat["uplink_ratio_last_5"] = float(
            sum(1 for d in directions if d == 1) / len(directions)) if directions else 0.0

        # ==== 加密特征 ====
        raw_data = bytes(pkt[Raw].load) if pkt.haslayer(Raw) else b""
        feat["entropy"] = float(calculate_entropy(raw_data))
        feat["chi_square"] = float(chi_square(raw_data))
        feat["printable_ratio"] = float(printable_ratio(raw_data))
        feat["null_byte_ratio"] = float(null_byte_ratio(raw_data))
        feat["byte_pair_corr"] = float(byte_pair_corr(raw_data))

        # ==== TLS特征 ====
        tls_info = parse_tls(pkt)
        feat["tls_record_type"] = int(tls_info["tls_record_type"])
        feat["tls_version"] = int(tls_info["tls_version"])
        feat["cipher_suite_len"] = int(tls_info["cipher_suite_len"])
        feat["handshake_phase"] = int(tls_info["handshake_phase"])

        if tls_info["handshake_phase"] == 3:
            key_update_count += 1
        feat["key_update_count"] = int(key_update_count)

        if last_handshake_time is not None:
            feat["time_since_last_handshake"] = float(current_time - last_handshake_time)
        else:
            feat["time_since_last_handshake"] = 0.0

        if tls_info["is_handshake"]:
            last_handshake_time = current_time

        # ==== 最终类型检查 ====
        feat = {k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v) if isinstance(v, (np.integer, int)) else v)
                for k, v in feat.items()}

        features.append(feat)

    return {
        "packet_features": features,
        "packet_payloads": payloads  # 新增返回payloads
    }

# === 提取流级特征和包级特征存入一个字典 ===
def extract_features(pcap_path):
    try:
        packets = rdpcap(pcap_path)
        # 新增条件：如果包数量不足5个，直接返回None
        if len(packets) < 5:
            return None
        # 获取 PCAP 文件所在目录的名称
        label = os.path.basename(os.path.dirname(os.path.dirname(pcap_path)))
        packet_result = extract_packet_features(packets)
        stat_result = extract_all_features(packets)
        return{
            "label": label,
            "stat_features": stat_result["stat_features"],
            "packet_features": packet_result["packet_features"],
            "payloads": packet_result["packet_payloads"]  # 新增返回payloads
        }

    except Exception as e:
        print(f"Error in {pcap_path}: {str(e)}")
        return None


# ==== 多进程主控部分 ====
def process_wrapper(args):
    pcap_path = args
    result = extract_features(pcap_path)
    if result is not None and result["stat_features"] != {} and result["packet_features"] != [] and result["payloads"] != []:
        # 添加文件名信息用于后续关联
        result["filename"] = os.path.basename(pcap_path)
    return result


# === 多进程提取特征 ===
def main(root_dir, output_feature_jsonl, output_payload_jsonl, num_workers):
    pool = multiprocessing.Pool(processes=num_workers)
    tasks = []

    # 任务收集逻辑保持不变
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            for subdir in os.listdir(label_dir):
                subdir_path = os.path.join(label_dir, subdir)
                if os.path.isdir(subdir_path):
                    for pcap_file in os.listdir(subdir_path):
                        if pcap_file.endswith('.pcap'):
                            pcap_path = os.path.join(subdir_path, pcap_file)
                            tasks.append(pcap_path)

    # 进度条处理保持不变
    with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
        results = []
        for result in pool.imap(process_wrapper, tasks):
            if result:  # 自动过滤返回None的无效结果
                results.append(result)
            pbar.update(1)

    # === 双文件写入逻辑 ===
    with open(output_feature_jsonl, 'w') as f_feature, \
            open(output_payload_jsonl, 'w') as f_payload:
        for result in results:
            # 特征文件写入
            feature_entry = {
                "filename": result["filename"],
                "label": result["label"],
                "stat_features": result["stat_features"],
                "packet_features":result["packet_features"]
            }
            json.dump(feature_entry, f_feature)
            f_feature.write('\n')

            # Payload文件写入
            payload_entry = {
                "filename": result["filename"],
                "label": result["label"],
                "payloads": result["payloads"]
            }
            json.dump(payload_entry, f_payload)
            f_payload.write('\n')

# ----------------------
# 数据加载
# ----------------------
def load_jsonl_data(file_path):
    """加载JSONL数据，返回特征序列列表和标签列表"""
    stat_features = []
    packet_sequenes = []
    all_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading JSONL data", unit="line"):
            data = json.loads(line)
            if "stat_features" in data and "packet_features" in data and 'label' in data:
                stat_features.append(data["stat_features"])
                packet_sequenes.append(data["packet_features"])
                all_labels.append(data['label'])
    return stat_features,packet_sequenes, all_labels

# ----------------------
# 特征预处理标准化
# ----------------------

def preprocess_time_steps(all_time_steps):
    """预处理所有时间步特征（统一缩放到 [0,1]）"""
    df = pd.DataFrame(all_time_steps)

    # 1. 处理 cipher_suite_len
    df['cipher_suite_len'] = df['cipher_suite_len'].clip(lower=0)

    # 2. 处理数值特征（不再用 log 变换）
    numeric_features = [
        'delta_time', 'relative_time', 'avg_delta_time_last_5',
        'time_since_last_handshake', 'packet_length', 'payload_length',
        'avg_pkt_len_last_5', 'std_pkt_len_last_5', 'uplink_ratio_last_5',
        'entropy', 'chi_square', 'printable_ratio', 'null_byte_ratio',
        'byte_pair_corr', 'window_size', 'seq_diff'
    ]

    # 统一用 MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))  # 或 (-1, 1)
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 3. 处理分类特征（用 OneHot 代替 factorize）
    tls_cat_features = ['tls_version', 'tls_record_type']
    # 将编码后的特征合并到 DataFrame
    print("处理分类特征为哑变量...")
    for col in ['tls_version', 'tls_record_type']:
        df[col], _ = pd.factorize(df[col].astype(str))

    # 检查 NaN
    assert not df.isna().any().any(), "存在 NaN 值！"

    return df.to_numpy(dtype=np.float32)

# ----------------------
# 序列填充（不足100的填充到100）
# ----------------------
def pad_sequences(sequences, max_len=100):
    """将变长序列填充到固定长度"""
    if not sequences:
        return np.zeros((0, max_len, 0), dtype=np.float32)

    feature_dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)

    for i, seq in enumerate(tqdm(sequences, desc=f"Padding sequences to {max_len}")):
        seq_len = min(len(seq), max_len)
        if seq_len > 0:
            padded[i, :seq_len] = seq[:seq_len]
    return padded

# ----------------------
# 主流程，整体标准化
# ----------------------
def Standard_dataset(input_path, output_path, max_len=100):
    try:
        time_features = rate_features = length_features = count_features = []

        # 1. 加载原始数据
        stat_features, packet_sequences, labels = load_jsonl_data(input_path)
        print("Label 分布：", np.unique(labels, return_counts=True))

        # 2. 展平所有时间步用于预处理
        print("展平所有时间步...")
        all_time_steps = []
        for seq in tqdm(packet_sequences, desc="Flattening time steps"):
            all_time_steps.extend(seq)

        # 3. 预处理所有时间步特征
        processed_steps = preprocess_time_steps(all_time_steps)


        # 4. 重新组装为样本序列
        print("重新组装每条序列...")
        processed_sequences = []
        pointer = 0
        for seq in tqdm(packet_sequences, desc="Reconstructing sequences"):
            seq_len = len(seq)
            processed_sequences.append(processed_steps[pointer:pointer + seq_len])
            pointer += seq_len

        # 5. 填充序列
        padded_sequences = pad_sequences(processed_sequences, max_len)

        # 6. 转换标签
        # labels, label_names = pd.factorize(labels)
        le = LabelEncoder()
        labels = le.fit_transform(labels)

        # 转换为DataFrame并显示进度
        print("\n正在将统计特征转换为DataFrame...")
        df = pd.DataFrame(stat_features)


        # 调试：打印 df.columns 查看实际列名
        print("\nDataFrame 列名：")
        print(df.columns)

        # 处理特征并显示进度
        print("\n正在处理特征...")

        # 指定特征分类
        time_features = [col for col in df.columns if 'IAT' in col or 'Duration' in col]
        length_features = [col for col in df.columns if 'Length' in col]
        rate_features = [col for col in df.columns if 'Bytes/s' in col or 'Packets/s' in col]
        count_features = [col for col in df.columns if
                          'Count' in col or 'Flag' in col or 'total Fwd' in col or "total Bwd" in col]
        count_features = list(set(count_features) - set(time_features) - set(length_features) - set(rate_features))

        print("  正在应用MinMax缩放...")
        scaler = MinMaxScaler(feature_range=(0, 1))  # 或 (-1, 1)
        all_features = time_features + rate_features + length_features + count_features

        if all_features:
            # 关键修改：直接MinMax，跳过log变换
            df[all_features] = scaler.fit_transform(df[all_features])

            # 调试检查
            print("  特征值范围验证:")
            print(f"  最小值: {df[all_features].min().min():.4f}")
            print(f"  最大值: {df[all_features].max().max():.4f}")

        stat_features = df.values

        # 7. 保存为NPZ
        print("保存到 NPZ 文件...")
        np.savez_compressed(output_path, stat_features=stat_features, sequences=padded_sequences, labels=labels)

        # 8. 打印统计信息
        print("\n✅ 处理完成！统计信息：")
        print(f"- 样本总数：{len(packet_sequences)}")
        print(f"- 填充后特征维度：{padded_sequences.shape}")
        print(f"- 标签分布：{dict(zip(*np.unique(labels, return_counts=True)))}")

    except Exception as e:
        print(f"发生错误：{e}")
        print(f"time_features: {time_features}")
        print(f"rate_features: {rate_features}")
        print(f"length_features: {length_features}")
        print(f"count_features: {count_features}")


if __name__ == "__main__":
    # ========== 主函数入口（示例） ==========
    # main("/path/to/your/root", "output.jsonl")

    # ========== 单一数据源路径示例 ==========
    # datapath_list = [
    #     "D:\\USTC-TFC2016-master\\Malware\\"
    # ]

    # ========== 多数据源配置（含注释说明，可按需开启） ==========
    # datapath_list = [
    #     # 合并后的NUDT移动数据集（未启用）
    #     # "D:\\NUDT_MobileTraffic\\merge\\"

    #     # USTC流量数据集（Benign 正常流量）
    #     "D:\\USTC-TFC2016-master\\Benign\\"

    #     # USTC流量数据集（Malware 恶意流量）
    #     # ,"D:\\USTC-TFC2016-master\\Malware\\"

    #     # ISCX VPN服务数据集
    #     , "D:\\ISCX-VPN-Service\\VPN\\"
    #     , "D:\\ISCX-VPN-Service\\NonVPN\\"

    #     # ISCX VPN应用数据集
    #     , "D:\\ISCX-VPN-App\\VPN\\"
    #     , "D:\\ISCX-VPN-App\\NonVPN\\"

    #     # ISCX Tor流量数据集
    #     , "D:\\ISCX-Tor\\Tor\\"
    #     , "D:\\ISCX-Tor\\NonTor\\"

    #     # CIC-IoT攻击数据（泛洪攻击与RTSP暴力破解）
    #     , "D:\\CIC_IOT_Dataset2022_Attacks\\Flood\\"
    #     , "D:\\CIC_IOT_Dataset2022_Attacks\\RTSP-Brute-Force\\"

    #     # 跨平台数据集（Android/iOS）
    #     , "D:\\CrossPlatform\\android\\"
    #     , "D:\\CrossPlatform\\ios\\"

    #     # Datacon2021 Part 1 数据集（样本与真实数据）
    #     , "D:\\datacon\\datacon2021_eta\\part1\\sample\\"
    #     , "D:\\datacon\\datacon2021_eta\\part1\\real_data\\"

    #     # Datacon2021 Part 2 数据集（训练与测试）
    #     , "D:\\datacon\\datacon2021_eta\\part2\\train_data\\"
    #     , "D:\\datacon\\datacon2021_eta\\part2\\train_data\\"

    #     # Datacon通用数据集
    #     , "D:\\datacon\\datacon_eta\\train\\"
    #     , "D:\\datacon\\datacon_eta\\test\\"
    # ]

    # ========== 当前使用的数据路径 ==========
    datapath_list = [
        "D:\\datacon\\datacon2021_eta\\part2\\train_data\\"
    ]
    for datapath in datapath_list:
        print(datapath)
        input = datapath + "\\splitcap\\"
        output = input + "all_features.jsonl"
        output_payload = input + "payload.jsonl"
        num_workers = 50
        # main(input, output, output_payload, num_workers)

        npz_path = input + "all_feature.npz"
        Standard_dataset(
            input_path=output,
            output_path=npz_path,
            max_len=100  # 设置序列最大长度
        )









