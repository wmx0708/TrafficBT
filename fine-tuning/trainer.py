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
    Monitor the training status of each component.
    model_dict: {"bert": bert_model, "trans": trans_model, "fusion": fusion_model}
    """
    print("\n" + "=" * 60)
    for name in ["bert", "trans", "fusion"]:
        model = model_dict[name]
        optim = optimizers[name]

        # Check the update ratio
        max_ratio = 0
        for param in model.parameters():
            if param.grad is not None:
                ratio = (param.grad.abs() * optim.param_groups[0]['lr']) / (param.abs() + 1e-6)
                max_ratio = max(max_ratio, ratio.max().item())

        # Check the momentum status
        mom_stats = []
        for param in model.parameters():
            if param in optim.state:
                mom_stats.append(optim.state[param]['exp_avg'].abs().mean().item())

        print(
            f"[{name.upper():<10}] "
            f"LR: {optim.param_groups[0]['lr']:.1e} | "
            f"Update Ratio: {max_ratio:.3e} | "
            f"Momentum Mean: {np.mean(mom_stats):.3e} | "
            f"Gradient Norm: {sum(p.grad.norm() for p in model.parameters() if p.grad is not None):.3e}"
        )
    print("=" * 60)


class ModalityAwareNormalizer:
    def __init__(self, modalities):
        """
        Improved multimodal gradient normalizer.
        Input format example:
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
        """Perform hierarchical gradient normalization."""
        for mod_name, config in self.modalities.items():
            params = config['params']
            target_norm = config['clip_val']

            # Get effective gradients
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                continue

            # Calculate the current modality's gradient norm
            current_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads])
            )

            # Avoid division by zero
            if current_norm < 1e-6:
                print(f"Warning: {mod_name} gradient norm is close to zero ({current_norm:.2e})")
                continue

            # Calculate the scaling factor and apply it
            scale = target_norm / current_norm
            for g in grads:
                g.mul_(scale)

            # Verify the normalization result
            new_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads])
            )
            print(f"{mod_name} gradient: {current_norm:.2e} → {new_norm:.2e} (target: {target_norm})")


class FlowTrainer:
    """Network traffic model trainer."""

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
        self.text_acc = 0.0  # Used to save the test set accuracy of the text model
        self.text_acc_threshold = 0.98
        self.cls_feat = []

        # Initialize SwanLab experiment
        self.swanlab = swanlab

    def train_epoch(self, loader, trans_train_loader):
        # Check if bert-only mode is adopted
        if self.text_acc > self.text_acc_threshold:
            if self.use_parallel:
                self.trans_model.module.warmup = True
            else:
                self.trans_model.warmup = True
            print("=== Using BERT-only mode for training (Fusion model not used) ===")
            self.text_model.train()  # Only train the text_model
            total_loss = 0
            all_preds, all_labels = [], []
            for batch_idx, batch1 in tqdm(enumerate(loader), desc="Training", total=len(loader)):
                inputs = {k: v.to(self.device) for k, v in batch1.items()}
                self.text_optim.zero_grad()

                # Text feature extraction
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
            self._log_metrics(metrics, prefix="train")  # Log training metrics

            return metrics, all_preds, all_labels, self.text_model, None, None
        else:
            if self.use_parallel:
                self.trans_model.module.warmup = False
                self.text_model.module.warmup = False
            else:
                self.trans_model.warmup = False
                self.text_model.warmup = False
            print("=== Training: Bert-Transformer ===")
            # Set model mode
            self.text_model.train()  # Still need train mode to ensure that some layers (like Dropout) work correctly
            self.trans_model.train()
            self.fusion_model.train()

            total_loss = 0
            all_preds, all_labels = [], []
            losses = []

            if self.use_parallel:
                # Define modality parameter groups
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
                # Define modality parameter groups
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
                # Data preparation
                inputs = {k: v.to(self.device) for k, v in batch1.items()}
                seq_x = batch2["sequence"].to(self.device)  # (B, T, D)
                stat_x = batch2["flow_feature"].to(self.device)  # (B, 1, D)
                labels = batch2["label"].to(self.device)

                # Zero out gradients (only for trainable parameters)
                self.text_optim.zero_grad()
                self.trans_optim.zero_grad()
                self.fusion_optim.zero_grad()

                # Text feature extraction (automatically skips gradient calculation for frozen layers)
                with torch.set_grad_enabled(True):  # Ensure local gradient calculation is allowed
                    cls_feat = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )

                # Transformer forward pass
                trans_feat = self.trans_model(stat_x, seq_x)

                # Fusion
                loss, final_output = self.fusion_model(cls_feat, trans_feat, inputs["label"])
                if self.use_parallel:
                    loss = loss.sum()
                # Backpropagation (only updates parameters that require gradients)
                loss.backward()
                # Gradient normalization
                normalizer.normalize()

                ### >>> Insert the first code snippet here (update ratio check) <<< ###
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

                # Gradient clipping (optional)
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

                # Parameter update
                self.text_optim.step()
                self.trans_optim.step()
                self.fusion_optim.step()
                self.text_scheduler.step()
                self.trans_scheduler.step()
                self.fusion_scheduler.step()

                if batch_idx % 50 == 0:
                    monitor_components(model, optimizers)

                # Log metrics
                total_loss += loss.item()
                all_preds.extend(final_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="train")  # Log training metrics

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
                    # Data preparation
                    inputs = {k: v.to(self.device) for k, v in batch1.items()}

                    bert_output = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )
                    # Fusion
                    loss = self.criterion(bert_output, inputs["label"])
                    if self.use_parallel:
                        loss = loss.sum()

                total_loss += loss.item()
                all_preds.extend(bert_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            self.validation_loss_history.append(total_loss / len(loader))
            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="eval")  # Log training metrics

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
                    # Data preparation
                    inputs = {k: v.to(self.device) for k, v in batch1.items()}
                    seq_x = batch2["sequence"].to(self.device)  # (B, T, D)
                    stat_x = batch2["flow_feature"].to(self.device)  # (B, 1, D)
                    labels = batch2["label"].to(self.device)

                    cls_feat = self.text_model(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )
                    trans_feat = self.trans_model(stat_x, seq_x)
                    # Fusion
                    loss, final_output = self.fusion_model(cls_feat, trans_feat, inputs["label"])
                    if self.use_parallel:
                        loss = loss.sum()

                    total_loss += loss.item()
                    all_preds.extend(final_output.argmax(dim=1).cpu().numpy())
                    all_labels.extend(inputs["label"].cpu().numpy())

                self.validation_loss_history.append(total_loss / len(loader))
                metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
                self._log_metrics(metrics, prefix="eval")  # Log training metrics

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
            # Check if the unfreezing conditions are met
            self.text_model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in tqdm(train_loader, desc=f"Text Epoch {epoch + 1}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                # BERT fine-tuning (automatically skips gradient calculation for frozen layers)
                with torch.set_grad_enabled(True):  # Ensure local gradient calculation is allowed
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

            # Testing
            val_total_loss = 0
            val_all_preds, val_all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Text Epoch {epoch + 1}"):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    # BERT fine-tuning (automatically skips gradient calculation for frozen layers)
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
            self._log_metrics(metrics, prefix="warmup_text")  # Log training metrics
            self._log_metrics(val_metrics, prefix="warmup_text_val")  # Log training metrics

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

                # Save predictions and true labels for metric calculation
                preds = logits.argmax(dim=1)  # Take the argmax as the predicted label
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Testing
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
            self._log_metrics(metrics, prefix="warmup_trans")  # Log training metrics
            self._log_metrics(val_metrics, prefix="warmup_trans_val")  # Log training metrics

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
            # If the text model accuracy exceeds 98%, return directly without warming up the fusion model
            print("Skipping Fusion Model warmup since text model accuracy exceeds 98%.")
            return
        else:
            self.fusion_model.train()
            self.text_model.eval()  # Freeze text_model
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
                self._log_metrics(metrics, prefix="warmup_fusion")  # Log training metrics

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
        """Uniformly log metrics to SwanLab."""
        log_data = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.swanlab.log(log_data)
