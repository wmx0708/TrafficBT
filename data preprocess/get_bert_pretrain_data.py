# 1. Import dependencies
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

# ==== Flow Statistical Feature Extraction Functions ====
def calculate_iat(timestamps):
    """Calculate statistical features for Inter-Arrival Time (IAT)"""
    if len(timestamps) < 2:
        return (0, 0, 0, 0)  # Returns a tuple of length 4
    iats = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    return (
        sum(iats) / len(iats),  # Mean
        statistics.stdev(iats) if len(iats) > 1 else 0,  # Std
        max(iats),  # Max
        min(iats),  # Min
        sum(iats)  # Total
    )


def calculate_packet_stats(lengths):
    """Calculate statistical features for packet lengths"""
    if not lengths:
        return (0, 0, 0, 0, 0)
    return (
        min(lengths),  # Min
        max(lengths),  # Max
        sum(lengths) / len(lengths),  # Mean
        statistics.stdev(lengths) if len(lengths) > 1 else 0,  # Std
        statistics.variance(lengths) if len(lengths) > 1 else 0  # Variance
    )

# ==== Packet Feature Calculation Functions ====
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

# ==== TLS Parsing Helper Functions ====
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


# ==== Time Feature Normalization Functions ====
def normalize_timestamp(timestamps):
    """Normalize timestamps (fixes numpy and Decimal conflict issue)"""
    # Convert to native Python float type
    timestamps_float = [float(ts) for ts in timestamps]

    # Calculate time range
    min_ts = min(timestamps_float)
    max_ts = max(timestamps_float)
    duration = max_ts - min_ts if max_ts > min_ts else 1.0  # Prevent division by zero

    # Normalization process
    return [(ts - min_ts) / duration for ts in timestamps_float]

 # === Convert payload to bigram format ===
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

# === Flow Statistical Feature Extraction Function ===
def extract_all_features(packets):
    """Main function: Extract features from a PCAP file"""
    if not packets:
        return {"stat_features": None}

    # Initialize statistical variables
    tcp_flags = defaultdict(int)
    fwd_packets = []
    bwd_packets = []
    all_packets = []
    flag_masks = {
        'F': 0x01, 'S': 0x02, 'R': 0x04, 'P': 0x08,
        'A': 0x10, 'U': 0x20, 'C': 0x80, 'E': 0x40
    }

    # Determine flow direction (based on the first packet)
    first_pkt = packets[0]
    src_ip = first_pkt[IP].src
    dst_ip = first_pkt[IP].dst
    sport = first_pkt[TCP].sport if TCP in first_pkt else first_pkt[UDP].sport
    dport = first_pkt[TCP].dport if TCP in first_pkt else first_pkt[UDP].dport

    # Process all packets
    for pkt in packets:
        if IP not in pkt:
            continue

        # Record TCP flags
        if TCP in pkt:
            flags = pkt[TCP].flags
            for flag, mask in flag_masks.items():
                if flags & mask:
                    tcp_flags[flag] += 1

        # Determine packet direction and record
        pkt_length = len(pkt)
        timestamp = float(pkt.time)
        if (pkt[IP].src, sport) == (src_ip, sport):  # Forward packet
            fwd_packets.append((timestamp, pkt_length))
        else:  # Backward packet
            bwd_packets.append((timestamp, pkt_length))
        all_packets.append((timestamp, pkt_length))

    # Sort by time
    all_packets.sort(key=lambda x: x[0])
    if not all_packets:
        return {"stat_features": None}

    # Calculate basic features
    start_time = all_packets[0][0]
    end_time = all_packets[-1][0]
    flow_duration = (end_time - start_time) * 1e6  # microseconds

    # Packet length statistics
    fwd_lengths = [p[1] for p in fwd_packets]
    bwd_lengths = [p[1] for p in bwd_packets]
    all_lengths = [p[1] for p in all_packets]

    # Inter-arrival time statistics
    fwd_times = [p[0] for p in fwd_packets]
    bwd_times = [p[0] for p in bwd_packets]
    all_times = [p[0] for p in all_packets]

    # Calculate statistics (assuming these functions are defined)
    fwd_stats = calculate_packet_stats(fwd_lengths)
    bwd_stats = calculate_packet_stats(bwd_lengths)
    all_stats = calculate_packet_stats(all_lengths)
    fwd_iat = calculate_iat(fwd_times)
    bwd_iat = calculate_iat(bwd_times)
    all_iat = calculate_iat(all_times)


    # Build feature dictionary
    features = {}

    # Basic information
    features['Flow Duration'] = flow_duration
    features['total Fwd Packet'] = len(fwd_packets)
    features['total Bwd packets'] = len(bwd_packets)
    features['total Length of Fwd Packet'] = sum(fwd_lengths)
    features['total Length of Bwd Packet'] = sum(bwd_lengths)

    # Forward packet statistics
    features['Fwd Packet Length Min'] = fwd_stats[0]
    features['Fwd Packet Length Max'] = fwd_stats[1]
    features['Fwd Packet Length Mean'] = fwd_stats[2]
    features['Fwd Packet Length Std'] = fwd_stats[3]

    # Backward packet statistics
    features['Bwd Packet Length Min'] = bwd_stats[0]
    features['Bwd Packet Length Max'] = bwd_stats[1]
    features['Bwd Packet Length Mean'] = bwd_stats[2]
    features['Bwd Packet Length Std'] = bwd_stats[3]

    # Flow rate
    features['Flow Bytes/s'] = (sum(all_lengths) / flow_duration) * 1e6 if flow_duration > 0 else 0
    features['Flow Packets/s'] = (len(all_packets) / flow_duration) * 1e6 if flow_duration > 0 else 0

    # IAT statistics
    features['Flow IAT Mean'] = all_iat[0]
    features['Flow IAT Std'] = all_iat[1]
    features['Flow IAT Max'] = all_iat[2]
    features['Flow IAT Min'] = all_iat[3]

    # Forward IAT
    features['Fwd IAT Mean']= fwd_iat[0]
    features['Fwd IAT Std']= fwd_iat[1] if len(fwd_iat) > 1 else 0
    features['Fwd IAT Max']= fwd_iat[2] if len(fwd_iat) > 2 else 0
    features['Fwd IAT Min']= fwd_iat[3] if len(fwd_iat) > 3 else 0
    features['Fwd IAT Total']= fwd_iat[4] if len(fwd_iat) > 4 else 0

    # Backward IAT
    features['Bwd IAT Mean']= bwd_iat[0]
    features['Bwd IAT Std']= bwd_iat[1] if len(bwd_iat) > 1 else 0
    features['Bwd IAT Max']= bwd_iat[2] if len(bwd_iat) > 2 else 0
    features['Bwd IAT Min']= bwd_iat[3] if len(bwd_iat) > 3 else 0
    features['Bwd IAT Total']= bwd_iat[4] if len(bwd_iat) > 4 else 0

    # TCP flags
    features['FIN Flag Count'] = tcp_flags.get('F', 0)
    features['SYN Flag Count'] = tcp_flags.get('S', 0)
    features['RST Flag Count'] = tcp_flags.get('R', 0)
    features['PSH Flag Count'] = tcp_flags.get('P', 0)
    features['ACK Flag Count'] = tcp_flags.get('A', 0)
    features['URG Flag Count'] = tcp_flags.get('U', 0)
    features['CWR Flag Count'] = tcp_flags.get('C', 0)
    features['ECE Flag Count'] = tcp_flags.get('E', 0)

    # Global packet statistics
    features['Packet Length Min'] = all_stats[0]
    features['Packet Length Max'] = all_stats[1]
    features['Packet Length Mean'] = all_stats[2]
    features['Packet Length Std'] = all_stats[3]
    features['Packet Length Variance'] = all_stats[4]

    return {
        "stat_features": features
    }


# ==== Packet-level Feature Extraction Function ====
def extract_packet_features(packets):
    features = []
    payloads = []  # New: Store payloads of the first 5 packets
    pkt_lens, deltas, directions = deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)

    # Consistently use float type for timestamps
    start_time = float(packets[0].time)
    prev_time = start_time
    prev_seq = None
    last_handshake_time = None
    client_ip = packets[0][IP].src if IP in packets[0] else None
    key_update_count = 0

    # Get and normalize timestamps
    timestamps = [float(pkt.time) for pkt in packets]
    normalized_timestamps = normalize_timestamp(timestamps)

    # New: Extract payloads of the first 5 packets
    for i in range(5):  # Ensure only the first 5 packets are processed
        pkt = packets[i]
        if bytes(pkt.payload):
            payload = binascii.hexlify(bytes(pkt.payload)).decode('utf-8')
            bigram_payload = bigram_generation(payload)
        payloads.append(bigram_payload)

    # Process features for all packets (maintain original logic)
    for i, pkt in enumerate(packets):
        feat = {}
        if not pkt.haslayer(IP):
            continue

    # Consistently use float type for timestamps
    start_time = float(packets[0].time)  # Convert to Python float
    prev_time = start_time
    prev_seq = None
    last_handshake_time = None
    client_ip = packets[0][IP].src if IP in packets[0] else None
    key_update_count = 0

    # Get and normalize timestamps
    timestamps = [float(pkt.time) for pkt in packets]  # Explicitly convert to float
    normalized_timestamps = normalize_timestamp(timestamps)

    for i, pkt in enumerate(packets):
        feat = {}
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]

        # ==== Time Features ====
        current_time = float(pkt.time)  # Ensure it is a Python float
        feat["timestamp"] = normalized_timestamps[i]
        feat["delta_time"] = current_time - prev_time
        feat["relative_time"] = current_time - start_time  # Use float operations directly
        prev_time = current_time

        # ==== Length and Direction Features ====
        plen = len(pkt)
        feat["packet_length"] = int(plen)  # Convert to Python int

        payload_len = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0
        feat["payload_length"] = int(payload_len)

        direction = 1 if ip.src == client_ip else -1
        feat["direction"] = int(direction)
        feat["is_ack_only"] = int(pkt.haslayer(TCP) and pkt[TCP].flags & 0x10 and payload_len == 0)

        # ==== Protocol Features ====
        feat["protocol_id"] = int(ip.proto)  # Convert to Python int

        # ==== TCP Features ====
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            flags = int(tcp.flags)  # Explicitly convert to int
            feat["tcp_flag_syn"] = int((flags & 0x02) != 0)
            feat["tcp_flag_ack"] = int((flags & 0x10) != 0)
            feat["tcp_flag_fin"] = int((flags & 0x01) != 0)

            # Handle sequence number difference
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

        # ==== Sliding Window Statistics ====
        pkt_lens.append(plen)
        deltas.append(feat["delta_time"])
        directions.append(direction)

        # Convert to native Python types
        feat["avg_pkt_len_last_5"] = float(np.mean(pkt_lens))
        feat["avg_delta_time_last_5"] = float(np.mean(deltas))
        feat["std_pkt_len_last_5"] = float(np.std(pkt_lens)) if len(pkt_lens) >= 2 else 0.0
        feat["uplink_ratio_last_5"] = float(
            sum(1 for d in directions if d == 1) / len(directions)) if directions else 0.0

        # ==== Encryption Features ====
        raw_data = bytes(pkt[Raw].load) if pkt.haslayer(Raw) else b""
        feat["entropy"] = float(calculate_entropy(raw_data))
        feat["chi_square"] = float(chi_square(raw_data))
        feat["printable_ratio"] = float(printable_ratio(raw_data))
        feat["null_byte_ratio"] = float(null_byte_ratio(raw_data))
        feat["byte_pair_corr"] = float(byte_pair_corr(raw_data))

        # ==== TLS Features ====
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

        # ==== Final Type Check ====
        feat = {k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v) if isinstance(v, (np.integer, int)) else v)
                for k, v in feat.items()}

        features.append(feat)

    return {
        "packet_features": features,
        "packet_payloads": payloads  # New: return payloads
    }

# === Extract flow-level and packet-level features into a dictionary ===
def extract_features(pcap_path):
    try:
        packets = rdpcap(pcap_path)
        # New condition: if packet count is less than 5, return None directly
        if len(packets) < 5:
            return None
        # Get the directory name of the PCAP file
        label = os.path.basename(os.path.dirname(os.path.dirname(pcap_path)))
        packet_result = extract_packet_features(packets)
        stat_result = extract_all_features(packets)
        return{
            "label": label,
            "stat_features": stat_result["stat_features"],
            "packet_features": packet_result["packet_features"],
            "payloads": packet_result["packet_payloads"]  # New: return payloads
        }

    except Exception as e:
        print(f"Error in {pcap_path}: {str(e)}")
        return None


# ==== Multiprocessing Control Section ====
def process_wrapper(args):
    pcap_path = args
    result = extract_features(pcap_path)
    if result is not None and result["stat_features"] != {} and result["packet_features"] != [] and result["payloads"] != []:
        # Add filename information for later association
        result["filename"] = os.path.basename(pcap_path)
    return result


# === Multiprocessing Feature Extraction ===
def main(root_dir, output_feature_jsonl, output_payload_jsonl, num_workers):
    pool = multiprocessing.Pool(processes=num_workers)
    tasks = []

    # Task collection logic remains unchanged
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

    # Progress bar handling remains unchanged
    with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
        results = []
        for result in pool.imap(process_wrapper, tasks):
            if result:  # Automatically filter out invalid results (None)
                results.append(result)
            pbar.update(1)

    # === Dual File Writing Logic ===
    with open(output_feature_jsonl, 'w') as f_feature, \
            open(output_payload_jsonl, 'w') as f_payload:
        for result in results:
            # Feature file writing
            feature_entry = {
                "filename": result["filename"],
                "label": result["label"],
                "stat_features": result["stat_features"],
                "packet_features":result["packet_features"]
            }
            json.dump(feature_entry, f_feature)
            f_feature.write('\n')

            # Payload file writing
            payload_entry = {
                "filename": result["filename"],
                "label": result["label"],
                "payloads": result["payloads"]
            }
            json.dump(payload_entry, f_payload)
            f_payload.write('\n')

# ----------------------
# Data Loading
# ----------------------
def load_jsonl_data(file_path):
    """Load JSONL data, return list of feature sequences and list of labels"""
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
# Feature Preprocessing and Standardization
# ----------------------

def preprocess_time_steps(all_time_steps):
    """Preprocess all time-step features (uniformly scale to [0,1])"""
    df = pd.DataFrame(all_time_steps)

    # 1. Process cipher_suite_len
    df['cipher_suite_len'] = df['cipher_suite_len'].clip(lower=0)

    # 2. Process numeric features (no longer using log transform)
    numeric_features = [
        'delta_time', 'relative_time', 'avg_delta_time_last_5',
        'time_since_last_handshake', 'packet_length', 'payload_length',
        'avg_pkt_len_last_5', 'std_pkt_len_last_5', 'uplink_ratio_last_5',
        'entropy', 'chi_square', 'printable_ratio', 'null_byte_ratio',
        'byte_pair_corr', 'window_size', 'seq_diff'
    ]

    # Use MinMaxScaler consistently
    scaler = MinMaxScaler(feature_range=(0, 1))  # or (-1, 1)
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 3. Process categorical features (using OneHot instead of factorize)
    tls_cat_features = ['tls_version', 'tls_record_type']
    # Merge encoded features into DataFrame
    print("Processing categorical features into dummy variables...")
    for col in ['tls_version', 'tls_record_type']:
        df[col], _ = pd.factorize(df[col].astype(str))

    # Check for NaN
    assert not df.isna().any().any(), "NaN values exist!"

    return df.to_numpy(dtype=np.float32)

# ----------------------
# Sequence Padding (pad to 100 if less than 100)
# ----------------------
def pad_sequences(sequences, max_len=100):
    """Pad variable-length sequences to a fixed length"""
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
# Main Process, Overall Standardization
# ----------------------
def Standard_dataset(input_path, output_path, max_len=100):
    try:
        time_features = rate_features = length_features = count_features = []

        # 1. Load raw data
        stat_features, packet_sequences, labels = load_jsonl_data(input_path)
        print("Label distribution:", np.unique(labels, return_counts=True))

        # 2. Flatten all time steps for preprocessing
        print("Flattening all time steps...")
        all_time_steps = []
        for seq in tqdm(packet_sequences, desc="Flattening time steps"):
            all_time_steps.extend(seq)

        # 3. Preprocess all time-step features
        processed_steps = preprocess_time_steps(all_time_steps)


        # 4. Reassemble into sample sequences
        print("Reassembling each sequence...")
        processed_sequences = []
        pointer = 0
        for seq in tqdm(packet_sequences, desc="Reconstructing sequences"):
            seq_len = len(seq)
            processed_sequences.append(processed_steps[pointer:pointer + seq_len])
            pointer += seq_len

        # 5. Pad sequences
        padded_sequences = pad_sequences(processed_sequences, max_len)

        # 6. Transform labels
        # labels, label_names = pd.factorize(labels)
        le = LabelEncoder()
        labels = le.fit_transform(labels)

        # Convert to DataFrame and show progress
        print("\nConverting statistical features to DataFrame...")
        df = pd.DataFrame(stat_features)


        # Debug: print df.columns to check actual column names
        print("\nDataFrame columns:")
        print(df.columns)

        # Process features and show progress
        print("\nProcessing features...")

        # Specify feature categories
        time_features = [col for col in df.columns if 'IAT' in col or 'Duration' in col]
        length_features = [col for col in df.columns if 'Length' in col]
        rate_features = [col for col in df.columns if 'Bytes/s' in col or 'Packets/s' in col]
        count_features = [col for col in df.columns if
                          'Count' in col or 'Flag' in col or 'total Fwd' in col or "total Bwd" in col]
        count_features = list(set(count_features) - set(time_features) - set(length_features) - set(rate_features))

        print("  Applying MinMax scaling...")
        scaler = MinMaxScaler(feature_range=(0, 1))  # or (-1, 1)
        all_features = time_features + rate_features + length_features + count_features

        if all_features:
            # Key change: direct MinMax, skip log transform
            df[all_features] = scaler.fit_transform(df[all_features])

            # Debug check
            print("  Feature value range validation:")
            print(f"  Minimum value: {df[all_features].min().min():.4f}")
            print(f"  Maximum value: {df[all_features].max().max():.4f}")

        stat_features = df.values

        # 7. Save to NPZ file
        print("Saving to NPZ file...")
        np.savez_compressed(output_path, stat_features=stat_features, sequences=padded_sequences, labels=labels)

        # 8. Print statistics
        print("\n✅ Processing complete! Statistics:")
        print(f"- Total number of samples: {len(packet_sequences)}")
        print(f"- Feature dimension after padding: {padded_sequences.shape}")
        print(f"- Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"time_features: {time_features}")
        print(f"rate_features: {rate_features}")
        print(f"length_features: {length_features}")
        print(f"count_features: {count_features}")


if __name__ == "__main__":
    # ========== Main function entry point (example) ==========
    # main("/path/to/your/root", "output.jsonl")

    # ========== Single data source path example ==========
    # datapath_list = [
    #     "D:\\USTC-TFC2016-master\\Malware\\"
    # ]

    # ========== Multi-data source configuration (with comments, can be enabled as needed) ==========
    # datapath_list = [
    #     # Merged NUDT mobile dataset (not enabled)
    #     # "D:\\NUDT_MobileTraffic\\merge\\"

    #     # USTC traffic dataset (Benign normal traffic)
    #     "D:\\USTC-TFC2016-master\\Benign\\"

    #     # USTC traffic dataset (Malware malicious traffic)
    #     # ,"D:\\USTC-TFC2016-master\\Malware\\"

    #     # ISCX VPN service dataset
    #     , "D:\\ISCX-VPN-Service\\VPN\\"
    #     , "D:\\ISCX-VPN-Service\\NonVPN\\"

    #     # ISCX VPN application dataset
    #     , "D:\\ISCX-VPN-App\\VPN\\"
    #     , "D:\\ISCX-VPN-App\\NonVPN\\"

    #     # ISCX Tor traffic dataset
    #     , "D:\\ISCX-Tor\\Tor\\"
    #     , "D:\\ISCX-Tor\\NonTor\\"

    #     # CIC-IoT attack data (Flood attack and RTSP Brute-Force)
    #     , "D:\\CIC_IOT_Dataset2022_Attacks\\Flood\\"
    #     , "D:\\CIC_IOT_Dataset2022_Attacks\\RTSP-Brute-Force\\"

    #     # Cross-platform dataset (Android/iOS)
    #     , "D:\\CrossPlatform\\android\\"
    -     , "D:\\CrossPlatform\\ios\\"

    #     # Datacon2021 Part 1 dataset (sample and real data)
    #     , "D:\\datacon\\datacon2021_eta\\part1\\sample\\"
    #     , "D:\\datacon\\datacon2021_eta\\part1\\real_data\\"

    #     # Datacon2021 Part 2 dataset (training and test)
    #     , "D:\\datacon\\datacon2021_eta\\part2\\train_data\\"
    #     , "D:\\datacon\\datacon2021_eta\\part2\\train_data\\"

    #     # Datacon general dataset
    #     , "D:\\datacon\\datacon_eta\\train\\"
    #     , "D:\\datacon\\datacon_eta\\test\\"
    # ]

    # ========== Currently used data path ==========
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
            max_len=100  # Set maximum sequence length
        )
