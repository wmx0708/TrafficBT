from transformers import BertForSequenceClassification, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextFeatureExtractor(nn.Module):
    """BERT文本特征提取器"""

    def __init__(self, pretrained_name: str, num_labels: int, use_selfclassifier_flag=False, dropout_prob: float = 0.3,
                 start_freeze_layer=11):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_name, hidden_dropout_prob=dropout_prob,
                                                 attention_probs_dropout_prob=dropout_prob)
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            output_hidden_states=True, ignore_mismatched_sizes=True)
        self.warmup = False
        # 是否使用自定义的分类器，比如以下的self.classifier就是我自定义的分类头
        self.use_selfclassifier_flag = use_selfclassifier_flag

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_labels)
        )
        # # 假设你冻结了模型的前几层，设置requires_grad=False
        for param in self.bert.bert.encoder.layer[:start_freeze_layer].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 遍历每个输入文本
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True)
        # 获取 [CLS] token 的最后一层隐藏状态
        cls_output = outputs.hidden_states[-1][:, 0, :]  # (batch_size, hidden_size)

        if self.warmup:
            if self.use_selfclassifier_flag:
                logits = self.classifier(self.dropout(cls_output))
                return self.softmax(logits)
            else:
                logits = outputs.logits
                return self.softmax(logits)
        else:
            return cls_output



class UnifiedFlowModel(nn.Module):
    def __init__(self, stat_feat_dim, stat_encoderseq_feat_dim=28, seq_len=100, hidden_dim=128, num_classes=100):
        super().__init__()

        # FlowTransformerClassifier 部分（统计特征）
        self.stat_embedding = nn.Linear(stat_feat_dim, hidden_dim)
        self.stat_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=0.1, batch_first=True),
            num_layers=2
        )

        # 定义位置编码（Positional Encoding），帮助Transformer了解序列中的顺序
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # 最大序列长度为1000
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # DualTransformer 部分（特征序列）
        self.feat_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=seq_feat_dim, nhead=4, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.proj = nn.Linear(seq_feat_dim, hidden_dim)
        self.seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=0.1, batch_first=True),
            num_layers=4
        )
        self.gate_heads = nn.ModuleList([nn.Linear(hidden_dim * 2, 1) for _ in range(3)])

        # 融合层和分类器
        self.dropout = nn.Dropout(0.5)

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def normalize_seq_x(self, seq_x):
        # 逐样本标准化（保持时序独立性）
        mean = seq_x.mean(dim=1, keepdim=True)  # [B, 1, F]
        std = seq_x.std(dim=1, keepdim=True)  # [B, 1, F]
        return (seq_x - mean) / (std + 1e-6)

    def forward(self, stat_x, seq_x):
        seq_x = self.normalize_seq_x(seq_x)
        # stat_x: (B, 1, stat_feat_dim)，统计特征
        # seq_x:  (B, T, seq_feat_dim=28)，特征序列

        # --- 统计特征编码 ---
        stat_feat = self.stat_embedding(stat_x)
        stat_feat = self.stat_encoder(stat_feat)
        stat_feat = stat_feat.mean(dim=1)  # [B, hidden_dim]

        # --- 特征序列编码 ---
        B, T, F = seq_x.shape
        seq_x = seq_x.reshape(B * T, F)
        seq_x = self.feat_encoder(seq_x)
        seq_x = seq_x.reshape(B, T, F)
        seq_feat = self.proj(seq_x)
        seq_feat = seq_feat + self.positional_encoding[:, :seq_feat.size(1), :]  # 加上位置编码
        seq_feat = self.seq_encoder(seq_feat)
        # seq_feat = seq_feat[:,-1,:]
        seq_feat = seq_feat.mean(dim=1)  # [B, d_model]

        # --- 融合 ---
        fusion = torch.cat([stat_feat, seq_feat], dim=1)  # [B, hidden_dim + d_model]
        fusion = self.fusion_proj(fusion)

        if self.warmup:
            out = self.classifier(fusion)
            return out
        else:
            return fusion


class FusionModel(nn.Module):
    def __init__(self, hidden_size=64, trans_hidden_size=64, num_classes=10, dropout=0.3):
        super(FusionModel, self).__init__()
        self.trans_hidden_size = trans_hidden_size
        self.hidden_size = hidden_size
        self.temperature = nn.Parameter(torch.tensor(0.5))  # 自动调节温度
        # CLS降维投影层
        self.cls_proj = nn.Linear(768, trans_hidden_size)
        self.trans_proj = nn.Linear(trans_hidden_size, trans_hidden_size)

        self.norm1 = nn.LayerNorm(self.trans_hidden_size)  # 对BERT特征
        self.norm2 = nn.LayerNorm(self.trans_hidden_size)  # 对Transformer特征

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.pre_fusion_dropout = nn.Dropout(dropout)

        # 使用多个门控头，避免单点失效
        self.gate_heads = nn.ModuleList([nn.Linear(trans_hidden_size * 2, 1) for _ in range(3)])
        self.gate_proj = nn.Sequential(
            nn.Linear(trans_hidden_size, trans_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trans_hidden_size, trans_hidden_size)
        )

        # # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(trans_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, bert_hidden, trans_hidden, labels=None):
        """
        bert_hidden: [B, T1, H]  - BERT的隐藏层输出（比如last_hidden_state）
        trans_hidden: [B, T2, H] - Transformer的隐藏层输出（特征序列）
        """

        # bert_hidden = self.dropout(self.cls_proj(bert_hidden))
        bert_hidden = self.relu(self.dropout(self.cls_proj(bert_hidden)))
        bert_hidden = self.norm1(bert_hidden)
        trans_hidden = self.relu(self.dropout(trans_hidden))
        # trans_hidden = self.dropout(trans_hidden)
        trans_hidden = self.norm2(trans_hidden)

        combined = torch.cat((bert_hidden, trans_hidden), dim=1)
        combined = self.pre_fusion_dropout(combined)

        # 每个头生成独立权重矩阵 [B, T, H]
        gate_logits = torch.stack([head(combined) for head in self.gate_heads], dim=-1)
        gate = torch.softmax(gate_logits, dim=-1)  # [B, 3]
        # 假设 gate 是形状为 (32, 1, 3) 的张量
        max_gate, _ = gate.max(dim=-1)  # 在最后一个维度选择最大值，_ 表示忽略索引

        fused_feats = max_gate * bert_hidden + (1 - max_gate) * trans_hidden
        fused_feats = self.gate_proj(fused_feats)

        logits = self.classifier(fused_feats)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        else:
            return logits
