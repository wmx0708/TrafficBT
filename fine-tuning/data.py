import os.path

from tqdm import tqdm
from transformers import BertTokenizer
from typing import List, Tuple, Dict
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.utils import resample
from collections import defaultdict


class NetworkFlowDataset(Dataset):
    """处理网络流量数据的Dataset类"""

    def __init__(self, data: List[Tuple], tokenizer: BertTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        texts, label = self.data[idx]
        concatenated_text = " ".join(texts)

        # 编码拼接后的文本
        encoding = self.tokenizer(
            concatenated_text,            
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # shape: [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class TransformerPacketDataset(Dataset):
    def __init__(self,data):
        self.X = torch.tensor(data["sequences"], dtype=torch.float32)
        self.y = torch.tensor(data["labels"], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "sequence": self.X[idx],  # shape: [max_length]
            "label":self.y[idx]
        }

class CombinedFlowDataset(Dataset):
    def __init__(self, flow_features,packet_sequences, labels):
        assert len(packet_sequences) == len(flow_features) == len(labels), "数据长度不一致"
        self.packet_sequences = torch.tensor(packet_sequences, dtype=torch.float32)
        self.flow_features = torch.tensor(flow_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        packet_seq = self.packet_sequences[idx]         # shape: [seq_len, feat_dim]
        flow_stat = self.flow_features[idx].unsqueeze(0) # shape: [1, stat_feat_dim]
        label = self.labels[idx]
        return {
            "sequence": packet_seq,      # 给 DualTransformer
            "flow_feature": flow_stat,   # 给 FlowTransformer
            "label": label
        }


# ==============Bert data加载===========================

def load_label_dict(data_path: str) -> Dict:
    if os.path.exists(f"{data_path}/splitcap/label_trans_dict.json"):
        """加载或创建标签字典"""
        with open(f"{data_path}/splitcap/label_trans_dict.json") as f:
            label_dict = json.load(f)
    else:
        label_set = set()
        with open(f"{data_path}/splitcap/tcn_payload.jsonl", 'r') as f:
            for line in tqdm(f, desc="读取中"):
                data = json.loads(line)
                if data['label'] != ".ipynb_checkpoints":
                    label_set.add(data['label'])
        # 构建 label -> id 映射
        label_dict = {label: idx for idx, label in enumerate(sorted(label_set))}
        with open(f"{data_path}/splitcap/label_dict.json", "w") as file:
            json.dump(label_dict, file)
    return label_dict


def load_flow_data(data_path: str, max_samples: int = 200) -> Tuple:
    """加载网络流量数据"""
    label_dict = load_label_dict(data_path)
    samples = []
    with open(f"{data_path}/splitcap/tcn_payload.jsonl", 'r') as f2:
        for line in tqdm(f2, desc="loading flow data......"):
            data = json.loads(line)
            if data["label"] == ".ipynb_checkpoints":
                continue
            packets = data["payloads"]
            samples.append((packets, label_dict[data["label"]]))
    return samples, list(label_dict.keys()), label_dict

# =====================Transformer数据增强==========================
# 局部噪声扰动函数
def add_noise(X_batch, noise_level=0.05):
    if isinstance(X_batch, np.ndarray):
        X_batch = torch.tensor(X_batch, dtype=torch.float32)  # 可选加 device=device
    noise = torch.randn_like(X_batch, device=X_batch.device) * noise_level
    # noise = np.random.normal(0, noise_level, X_batch.shape,device=X_batch.device)
    return X_batch + noise

# 掩码数据包函数（丢包）
def mask_packets(X_batch, mask_prob=0.1):
    batch_size, seq_len, feature_len = X_batch.shape
    for i in range(batch_size):
        # 随机删除一些数据包
        num_masked = int(seq_len * mask_prob)
        masked_indices = np.random.choice(seq_len, num_masked, replace=False)
        X_batch[i, masked_indices] = 0  # 用0替换丢失的包
    return X_batch

# 顺序扰动函数
def shuffle_packets(X_batch, shuffle_prob=0.1):
    batch_size, seq_len, feature_len = X_batch.shape
    for i in range(batch_size):
        if torch.rand(1).item() < shuffle_prob:  # 每个batch可能进行扰动
            indices = torch.randperm(seq_len)  # 生成随机索引
            X_batch[i] = X_batch[i][indices]  # 按随机索引打乱数据包顺序
    return X_batch

# 数据增强：将噪声扰动、丢包和顺序扰动结合
def augment_data(X_batch, noise_level=0.05, mask_prob=0.1, shuffle_prob=0.1):
    X_batch = add_noise(X_batch, noise_level)
    X_batch = mask_packets(X_batch, mask_prob)
    X_batch = shuffle_packets(X_batch, shuffle_prob)
    return X_batch

# 过采样并进行数据增强
def balance_and_augment_data(X, y, target_class_count, noise_level=0.05, mask_prob=0.1, shuffle_prob=0.1):
    # 获取每个类别的样本数量
    unique_classes = np.unique(y)
    balanced_X = []
    balanced_y = []

    for class_label in unique_classes:
        # 获取该类别的所有样本
        class_X = X[y == class_label]
        class_y = y[y == class_label]

        # 如果该类别样本数少于目标数量，进行过采样
        if len(class_X) < target_class_count:
            # 过采样，填充至目标数量
            class_X_resampled, class_y_resampled = resample(class_X, class_y,
                                                            replace=True,
                                                            n_samples=target_class_count,
                                                            random_state=42)
        else:
            # 如果该类别样本数多于目标数量，截取多余的部分
            class_X_resampled = class_X[:target_class_count]
            class_y_resampled = class_y[:target_class_count]

        # 对该类别的所有样本应用数据增强
        class_X_resampled = augment_data(class_X_resampled, noise_level, mask_prob, shuffle_prob)

        # 将增强后的数据加入平衡后的数据集中
        balanced_X.append(class_X_resampled)
        balanced_y.append(class_y_resampled)

    # 合并所有类别的样本
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    return balanced_X, balanced_y

def  get_balanced_trans_dataset(trans_dataset, target_class_count=1000,
                                noise_level=0.05, mask_prob=0.1, shuffle_prob=0.1):
    """
    输入原始 trans_dataset，返回增强后、平衡的 Dataset。
    """
    # 提取原始数据
    X = trans_dataset.X.numpy()
    y = trans_dataset.y.numpy()

    # 收集每个类别的数据
    unique_classes = np.unique(y)
    balanced_X = []
    balanced_y = []

    for cls in unique_classes:
        cls_X = X[y == cls]
        cls_y = y[y == cls]

        if len(cls_X) < target_class_count:
            cls_X_resampled, cls_y_resampled = resample(
                cls_X, cls_y, replace=True,
                n_samples=target_class_count, random_state=42)
        else:
            cls_X_resampled = cls_X[:target_class_count]
            cls_y_resampled = cls_y[:target_class_count]

        # 数据增强
        cls_X_resampled = augment_data(cls_X_resampled, noise_level, mask_prob, shuffle_prob)

        balanced_X.append(cls_X_resampled)
        balanced_y.append(cls_y_resampled)

    # 合并所有类别数据
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    # 打包为新 dataset
    balanced_data = {
        "sequences": balanced_X,
        "labels": balanced_y
    }
    print(f"balanced all features to {target_class_count}")
    return TransformerPacketDataset(balanced_data)

def augment_sample(sample, noise_level=0.01):
    """为样本添加高斯噪声"""
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32)  # 可选加 device=device
    noise = torch.randn_like(sample, device=sample.device) * noise_level
    # noise = np.random.normal(0, noise_level, size=sample.shape,device=sample.device)
    return sample + noise

def augment_and_balance(features, labels, target_count=500, noise_level=0.01):
    """
    对每个类别进行过采样或欠采样到 target_count，大部分通过高斯噪声增强实现
    """
    new_features = []
    new_labels = []

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    for label, indices in label_to_indices.items():
        current_count = len(indices)
        label_features = features[indices]

        if current_count < target_count:
            # 过采样：复制并添加高斯噪声
            n_to_add = target_count - current_count
            repeats = n_to_add // current_count
            remainder = n_to_add % current_count

            # 扩展已有样本
            for _ in range(repeats):
                for sample in label_features:
                    new_features.append(augment_sample(sample, noise_level))
                    new_labels.append(label)
            for i in range(remainder):
                new_features.append(augment_sample(label_features[i], noise_level))
                new_labels.append(label)

            # 加入原始样本
            new_features.extend(label_features)
            new_labels.extend([label] * current_count)

        elif current_count > target_count:
            # 欠采样：随机选择 target_count 个样本
            selected_idx = np.random.choice(indices, target_count, replace=False)
            new_features.extend(features[selected_idx])
            new_labels.extend(labels[selected_idx])

        else:
            # 恰好等于目标数量，直接加入
            new_features.extend(label_features)
            new_labels.extend([label] * current_count)

    return np.array(new_features), np.array(new_labels)

# ------------------ 掩码增强 ------------------

def random_token_masking(texts: List[str], mask_prob=0.15, mask_token="[MASK]") -> List[str]:
    """对文本列表中的每个字符串进行 token-level 随机掩码增强"""
    masked = []
    for payload in texts:
        tokens = payload.split()  # 注意：这里假设 payload 是空格分词的
        new_tokens = [
            mask_token if random.random() < mask_prob else t for t in tokens
        ]
        masked.append(" ".join(new_tokens))
    return masked

# ------------------ 主函数 ------------------

def get_balanced_masked_bert_dataset(samples: List[Tuple[List[str], int]],
                                     tokenizer: BertTokenizer,
                                     target_count: int,
                                     mask_prob: float = 0.15,
                                     max_length: int = 512) -> NetworkFlowDataset:
    """
    对输入的 BERT 文本样本做标签平衡，并在过采样时进行掩码增强。
    返回 NetworkFlowDataset。
    """
    from collections import defaultdict
    label_to_samples = defaultdict(list)
    for s in samples:
        label_to_samples[s[1]].append(s)

    new_samples = []

    for label, s_list in label_to_samples.items():
        if len(s_list) >= target_count:
            # 截断
            selected = s_list[:target_count]
        else:
            # 重采样+增强
            selected = []
            needed = target_count - len(s_list)
            repeats = resample(s_list, n_samples=needed, replace=True, random_state=42)
            for orig_texts, _ in repeats:
                masked_texts = random_token_masking(orig_texts, mask_prob)
                selected.append((masked_texts, label))
            # 添加原始数据
            selected += s_list

        new_samples.extend(selected)
    print(f"Resampl samples to {target_count}")
    return new_samples
    # return NetworkFlowDataset(new_samples, tokenizer=tokenizer, max_length=max_length)