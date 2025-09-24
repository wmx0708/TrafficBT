import copy
import swanlab
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def monitor_components(model_dict, optimizers):
    """
    监控各组件训练状态
    model_dict: {"bert": bert_model, "trans": trans_model, "fusion": fusion_model}
    """
    print("\n" + "=" * 60)
    for name in ["bert", "trans", "fusion"]:
        model = model_dict[name]
        optim = optimizers[name]

        # 检查更新比率
        max_ratio = 0
        for param in model.parameters():
            if param.grad is not None:
                ratio = (param.grad.abs() * optim.param_groups[0]['lr']) / (param.abs() + 1e-6)
                max_ratio = max(max_ratio, ratio.max().item())

        # 检查动量状态
        mom_stats = []
        for param in model.parameters():
            if param in optim.state:
                mom_stats.append(optim.state[param]['exp_avg'].abs().mean().item())

        print(
            f"[{name.upper():<10}] "
            f"LR: {optim.param_groups[0]['lr']:.1e} | "
            f"更新比率: {max_ratio:.3e} | "
            f"动量均值: {np.mean(mom_stats):.3e} | "
            f"梯度范数: {sum(p.grad.norm() for p in model.parameters() if p.grad is not None):.3e}"
        )
    print("=" * 60)


class ModalityAwareNormalizer:
    def __init__(self, modalities):
        """
        改进版多模态梯度归一化器
        输入格式示例：
        modalities = {
            'text': {
                'params': text_model.parameters(),
                'clip_val': 0.1
            },
            'trans': {
                'params': trans_model.parameters(),
                'clip_val': 1.0
            },
            'fusion': {
                'params': fusion_model.parameters(),
                'clip_val': 0.5
            }
        }
        """
        self.modalities = modalities

    def normalize(self):
        """执行分层梯度归一化"""
        for mod_name, config in self.modalities.items():
            params = config['params']
            target_norm = config['clip_val']

            # 获取有效梯度
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                continue

            # 计算当前模态梯度范数
            current_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads])
            )

            # 避免除零
            if current_norm < 1e-6:
                print(f"警告: {mod_name}梯度范数接近零 ({current_norm:.2e})")
                continue

            # 计算缩放因子并应用
            scale = target_norm / current_norm
            for g in grads:
                g.mul_(scale)

            # 验证归一化结果
            new_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads])
            )
            print(f"{mod_name}梯度: {current_norm:.2e} → {new_norm:.2e} (目标: {target_norm})")


class FlowTrainer:
    """网络流量模型训练器"""

    def __init__(self, num_classes, text_model, trans_model, fusion_model, text_optim, trans_optim, fusion_optim,
                 text_scheduler, trans_scheduler, fusion_scheduler, device, use_parallel, start_freeze_layer, swanlab):
        self.num_classes = num_classes
        self.text_model = text_model
        self.trans_model = trans_model
        self.fusion_model = fusion_model
        self.text_optim = text_optim
        self.trans_optim = trans_optim
        self.fusion_optim = fusion_optim
        self.text_scheduler = text_scheduler
        self.trans_scheduler = trans_scheduler
        self.fusion_scheduler = fusion_scheduler
        self.lora_warmup_epochs = 5
        self.device = device
        self.use_parallel = use_parallel
        self.current_layer = start_freeze_layer
        self.validation_loss_history = []
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.text_acc = 0.0  # 用于保存文本模型的测试集准确率
        self.text_acc_threshold = 0.98
        self.cls_feat = []

        # 初始化SwanLab实验
        self.swanlab = swanlab

    def train_epoch(self, loader, trans_train_loader):
        # 检查是否采用bert-only模式
        if self.text_acc > self.text_acc_threshold:
            if self.use_parallel:
                self.trans_model.module.warmup = True
            else:
                self.trans_model.warmup = True
            print("=== Using BERT-only mode for training (Fusion model not used) ===")
            self.text_model.train()  # 只训练text_model
            total_loss = 0
            all_preds, all_labels = [], []
            for batch_idx, batch1 in tqdm(enumerate(loader), desc="Training", total=len(loader)):
                inputs = {k: v.to(self.device) for k, v in batch1.items()}
                self.text_optim.zero_grad()

                # 文本特征提取
                with torch.set_grad_enabled(True):
                    bert_output = self.text_model(inputs["input_ids"], inputs["attention_mask"])

                loss = self.criterion(bert_output, inputs["label"])
                if self.use_parallel:
                    loss = loss.sum()
                loss.backward()

                if self.use_parallel:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.text_model.module.parameters()),
                        max_norm=10.0
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.text_model.parameters()),
                        max_norm=10.0
                    )

                self.text_optim.step()
                self.text_scheduler.step()

                total_loss += loss.item()
                all_preds.extend(bert_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="train")  # 记录训练指标

            return metrics, all_preds, all_labels, self.text_model, None, None
        else:
            if self.use_parallel:
                self.trans_model.module.warmup = False
                self.text_model.module.warmup = False
            else:
                self.trans_model.warmup = False
                self.text_model.warmup = False
            print("=== Training: Bert-Transformer ===")
            # 设置模型模式
            self.text_model.train()  # 仍然需要train模式以保证部分层（如Dropout）正常工作
            self.trans_model.train()
            self.fusion_model.train()

            total_loss = 0
            all_preds, all_labels = [], []
            losses = []

            if self.use_parallel:
                # 定义模态参数组
                modalities = {
                    'text': {
                        'params': self.text_model.module.parameters(),
                        'clip_val': 0.1
                    },
                    'trans': {
                        'params': self.trans_model.module.parameters(),
                        'clip_val': 1.0
                    },
                    'fusion': {
                        'params': self.fusion_model.module.parameters(),
                        'clip_val': 0.5
                    }
                }
            else:
                # 定义模态参数组
                modalities = {
                    'text': {
                        'params': self.text_model.parameters(),
                        'clip_val': 0.1
                    },
                    'trans': {
                        'params': self.trans_model.parameters(),
                        'clip_val': 1.0
                    },
                    'fusion': {
                        'params': self.fusion_model.parameters(),
                        'clip_val': 1.0
                    }
                }

            normalizer = ModalityAwareNormalizer(modalities)

            for batch_idx, (batch1, batch2) in tqdm(enumerate(zip(loader, trans_train_loader)), desc="Training",
                                                    total=len(loader)):
                # 数据准备
                inputs = {k: v.to(self.device) for k, v in batch1.items()}
                seq_x = batch2["sequence"].to(self.device)  # (B, T, D)
                stat_x = batch2["flow_feature"].to(self.device)  # (B, 1, D)
                labels = batch2["label"].to(self.device)

                # 梯度清零（只更新有效参数）
                self.text_optim.zero_grad()
                self.trans_optim.zero_grad()
                self.fusion_optim.zero_grad()

                # 文本特征提取（自动跳过冻结层计算梯度）
                with torch.set_grad_enabled(True):  # 确保允许局部梯度计算
                    cls_feat = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )

                # Transformer前向
                trans_feat = self.trans_model(stat_x, seq_x)

                # 融合
                loss, final_output = self.fusion_model(cls_feat, trans_feat, inputs["label"])
                if self.use_parallel:
                    loss = loss.sum()
                # 反向传播（仅更新需要梯度的参数）
                loss.backward()
                # 梯度归一化
                normalizer.normalize()

                ### >>> 在这里插入第一段代码（更新比率检查）<<< ###
                optimizers = {
                    "bert": self.text_optim,
                    "trans": self.trans_optim,
                    "fusion": self.fusion_optim
                }
                model = {
                    "bert": self.text_model,
                    "trans": self.trans_model,
                    "fusion": self.fusion_model
                }
                if batch_idx % 10 == 0:
                    monitor_components(model, optimizers)

                # 梯度裁剪（可选）
                if self.use_parallel:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.text_model.module.parameters()),
                        max_norm=5.0
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.text_model.parameters()),
                        max_norm=5.0
                    )
                if self.use_parallel:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.fusion_model.parameters()),
                        max_norm=1.0
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.fusion_model.parameters()),
                        max_norm=1.0
                    )

                # 参数更新
                self.text_optim.step()
                self.trans_optim.step()
                self.fusion_optim.step()
                self.text_scheduler.step()
                self.trans_scheduler.step()
                self.fusion_scheduler.step()

                if batch_idx % 50 == 0:
                    monitor_components(model, optimizers)

                # 记录指标
                total_loss += loss.item()
                all_preds.extend(final_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="train")  # 记录训练指标

            return metrics, all_preds, all_labels, self.text_model, self.trans_model, self.fusion_model

    def evaluate(self, loader, trans_loader):
        if self.text_acc > self.text_acc_threshold:
            print("=== Using BERT-only mode for eval (Fusion model not used) ===")
            self.text_model.eval()
            total_loss = 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch_idx, (batch1, batch2) in tqdm(enumerate(zip(loader, trans_loader)), desc="Evaluating",
                                                        total=len(loader)):
                    # 数据准备
                    inputs = {k: v.to(self.device) for k, v in batch1.items()}

                    bert_output = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )
                    # 融合
                    loss = self.criterion(bert_output, inputs["label"])
                    if self.use_parallel:
                        loss = loss.sum()

                total_loss += loss.item()
                all_preds.extend(bert_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            self.validation_loss_history.append(total_loss / len(loader))
            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="eval")  # 记录训练指标

            return metrics, all_preds, all_labels
        else:
            print("=== Evaluating: Bert-Transformer ===")
            self.text_model.eval()
            self.trans_model.eval()
            self.fusion_model.eval()

            total_loss = 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch_idx, (batch1, batch2) in tqdm(enumerate(zip(loader, trans_loader)), desc="Evaluating",
                                                        total=len(loader)):
                    # 数据准备
                    inputs = {k: v.to(self.device) for k, v in batch1.items()}
                    seq_x = batch2["sequence"].to(self.device)  # (B, T, D)
                    stat_x = batch2["flow_feature"].to(self.device)  # (B, 1, D)
                    labels = batch2["label"].to(self.device)

                    cls_feat = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )
                    trans_feat = self.trans_model(stat_x, seq_x)
                    # 融合
                    loss, final_output = self.fusion_model(cls_feat, trans_feat, inputs["label"])
                    if self.use_parallel:
                        loss = loss.sum()

                    total_loss += loss.item()
                    all_preds.extend(final_output.argmax(dim=1).cpu().numpy())
                    all_labels.extend(inputs["label"].cpu().numpy())

                self.validation_loss_history.append(total_loss / len(loader))
                metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
                self._log_metrics(metrics, prefix="eval")  # 记录训练指标

            return metrics, all_preds, all_labels

    def warmup_text_model(self, train_loader, max_epochs=5, val_loader=None, patience=2):
        print("=== Warmup: Bert ===")
        if self.use_parallel:
            self.text_model.module.warmup = True
        else:
            self.text_model.warmup = True
        best_acc = 0.0
        epochs_no_improve = 0
        best_state = None
        validation_loss_history = []
        current_layer = self.current_layer

        for epoch in range(max_epochs):
            # 检查是否满足解冻条件
            self.text_model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in tqdm(train_loader, desc=f"Text Epoch {epoch + 1}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                # bert微调（自动跳过冻结层计算梯度）
                with torch.set_grad_enabled(True):  # 确保允许局部梯度计算
                    output = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )
                self.text_optim.zero_grad()

                loss = self.criterion(output, inputs["label"])
                if self.use_parallel:
                    loss = loss.sum()
                loss.backward()
                self.text_optim.step()
                self.text_scheduler.step()

                total_loss += loss.item()
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            # 测试
            val_total_loss = 0
            val_all_preds, val_all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Text Epoch {epoch + 1}"):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    # bert微调（自动跳过冻结层计算梯度）
                    output = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"])

                    loss = self.criterion(output, inputs["label"])
                    if self.use_parallel:
                        loss = loss.sum()

                    val_total_loss += loss.item()
                    val_all_preds.extend(output.argmax(dim=1).cpu().numpy())
                    val_all_labels.extend(inputs["label"].cpu().numpy())

            validation_loss_history.append(val_total_loss / len(val_loader))
            acc = accuracy_score(all_labels, all_preds)
            val_acc = accuracy_score(val_all_labels, val_all_preds)
            print(f"Epoch {epoch + 1}: Text Loss = {total_loss / len(train_loader):.4f}, Acc = {acc:.4f}")
            print(f"Epoch {epoch + 1}: Text Val Loss = {val_total_loss / len(val_loader):.4f}, Val Acc = {val_acc:.4f}")
            metrics = self._compute_metrics(total_loss / len(train_loader), all_preds, all_labels)
            val_metrics = self._compute_metrics(val_total_loss / len(val_loader), val_all_preds, val_all_labels)
            self._log_metrics(metrics, prefix="warmup_text")  # 记录训练指标
            self._log_metrics(val_metrics, prefix="warmup_text_val")  # 记录训练指标

            if val_acc > best_acc:
                best_acc = val_acc
                self.text_acc = best_acc
                epochs_no_improve = 0
                if self.use_parallel:
                    best_state = copy.deepcopy(self.text_model.state_dict())
                else:
                    best_state = copy.deepcopy(self.text_model.state_dict())
            else:
                epochs_no_improve += 1

        if best_state:
            self.text_model.load_state_dict(best_state)

    def warmup_trans_model(self, train_loader, max_epochs=5, val_loader=None, patience=2):
        print("=== Warmup: TransModel ===")
        if self.use_parallel:
            self.trans_model.module.warmup = True
        else:
            self.trans_model.warmup = True

        best_acc = 0.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            self.trans_model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in tqdm(train_loader, desc=f"Trans Epoch {epoch + 1}"):
                seq_x = batch["sequence"].to(self.device)  # (B, T, D)
                stat_x = batch["flow_feature"].to(self.device)  # (B, 1, D)
                labels = batch["label"].to(self.device)

                logits = self.trans_model(stat_x, seq_x)
                loss = self.criterion(logits, labels)
                if self.use_parallel:
                    loss = loss.sum()

                self.trans_optim.zero_grad()
                loss.backward()
                self.trans_optim.step()
                self.trans_scheduler.step()

                total_loss += loss.item()

                # 保存预测值和真实标签，用于计算指标
                preds = logits.argmax(dim=1)  # 取最大值作为预测标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 测试
            self.trans_model.eval()
            val_total_loss = 0
            val_all_preds, val_all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Trans Epoch {epoch + 1}"):
                    seq_x = batch["sequence"].to(self.device)  # (B, T, D)
                    stat_x = batch["flow_feature"].to(self.device)  # (B, 1, D)
                    labels = batch["label"].to(self.device)

                    logits = self.trans_model(stat_x, seq_x)
                    loss = self.criterion(logits, labels)
                    if self.use_parallel:
                        loss = loss.sum()

                    val_total_loss += loss.item()
                    val_all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            val_acc = accuracy_score(val_all_labels, val_all_preds)
            print(f"Epoch {epoch + 1}: Trans Loss = {total_loss:.4f}, Acc = {acc:.4f}")
            print(f"Epoch {epoch + 1}: Trans Val Loss = {val_total_loss:.4f}, Val Acc = {val_acc:.4f}")
            metrics = self._compute_metrics(total_loss / len(train_loader), all_preds, all_labels)
            val_metrics = self._compute_metrics(val_total_loss / len(val_loader), val_all_preds, val_all_labels)
            self._log_metrics(metrics, prefix="warmup_trans")  # 记录训练指标
            self._log_metrics(val_metrics, prefix="warmup_trans_val")  # 记录训练指标

            if acc > best_acc:
                best_acc = acc
                epochs_no_improve = 0
                if self.use_parallel:
                    best_state = copy.deepcopy(self.trans_model.module.state_dict())
                else:
                    best_state = copy.deepcopy(self.trans_model.state_dict())
            else:
                epochs_no_improve += 1

        if best_state:
            if self.use_parallel:
                self.trans_model.module.load_state_dict(best_state)
            else:
                self.trans_model.load_state_dict(best_state)

    def warmup_fusion_model(self, text_loader, trans_loader, max_epochs=5, patience=2):
        if self.text_acc > self.text_acc_threshold:
            print("Skipping Fusion Model warmup since text model accuracy exceeds 98%.")
            return  # 如果文本模型准确率超过98%，直接返回，不进行融合模型的warmup
        else:
            self.fusion_model.train()
            self.text_model.eval()  # 冻结 text_model
            self.trans_model.eval()
            print("=== Warmup: FusionModel ===")
            if self.use_parallel:
                self.trans_model.module.warmup = False
                self.text_model.module.warmup = False
            else:
                self.trans_model.warmup = False
                self.text_model.warmup = False
            best_acc = 0.0
            epochs_no_improve = 0
            best_state = None

            for epoch in range(max_epochs):
                total_loss = 0
                all_preds, all_labels = [], []

                for batch_idx, (batch1, batch2) in tqdm(enumerate(zip(text_loader, trans_loader)),
                                                        desc=f"Fusion Epoch {epoch + 1}", total=len(text_loader)):
                    inputs = {k: v.to(self.device) for k, v in batch1.items()}
                    seq_x = batch2["sequence"].to(self.device)  # (B, T, D)
                    stat_x = batch2["flow_feature"].to(self.device)  # (B, 1, D)
                    labels = batch2["label"].to(self.device)

                    self.fusion_optim.zero_grad()

                    if len(self.cls_feat) < len(text_loader):
                        with torch.no_grad():
                            cls_feat = self.text_model(inputs["input_ids"], inputs["attention_mask"])
                            self.cls_feat.append(cls_feat.to("cpu"))
                    else:
                        cls_feat = self.cls_feat[batch_idx].to(self.device)
                    trans_feat = self.trans_model(stat_x, seq_x)

                    loss, final_output = self.fusion_model(cls_feat, trans_feat, inputs["label"])
                    if self.use_parallel:
                        loss = loss.sum()
                    loss.backward()
                    self.fusion_optim.step()
                    self.fusion_scheduler.step()

                    total_loss += loss.item()
                    all_preds.extend(final_output.argmax(dim=1).cpu().numpy())
                    all_labels.extend(inputs["label"].cpu().numpy())
                acc = accuracy_score(all_labels, all_preds)
                print(f"Epoch {epoch + 1}/{max_epochs}: Fusion Loss = {total_loss:.4f}, Acc = {acc:.4f}")
                metrics = self._compute_metrics(total_loss / len(text_loader), all_preds, all_labels)
                self._log_metrics(metrics, prefix="warmup_fusion")  # 记录训练指标

                if acc > best_acc:
                    best_acc = acc
                    epochs_no_improve = 0
                    if self.use_parallel:
                        best_state = copy.deepcopy(self.fusion_model.state_dict())
                    else:
                        best_state = copy.deepcopy(self.fusion_model.state_dict())
                else:
                    epochs_no_improve += 1

            if best_state:
                if self.use_parallel:
                    self.fusion_model.load_state_dict(best_state)
                else:
                    self.fusion_model.load_state_dict(best_state)

    def _compute_metrics(self, avg_loss, preds, labels):
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0)
        }
        return metrics

    def _log_metrics(self, metrics: dict, prefix: str = "train"):
        """统一记录指标到SwanLab"""
        log_data = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.swanlab.log(log_data)