import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    TensorDataset,
    WeightedRandomSampler,
)
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


torch.set_num_threads(5)
torch.set_num_interop_threads(5)


LABEL2ID = {"contradiction": 0, "neutral": 1, "entailment": 2}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|n't|never|none|nothing|nobody|neither|nowhere|hardly|scarcely|barely|"
    r"cannot|can't|without|won't|doesn't|don't|didn't|isn't|aren't|couldn't|shouldn't|"
    r"wouldn't|wasn't|weren't|hasn't|haven't|hadn't)\b",
    flags=re.IGNORECASE,
)


def seed_everything(seed: int) -> None:
    """设置随机种子，确保结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def contains_negation(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(NEGATION_PATTERN.search(text))


def detect_negation(premise: str, hypothesis: str) -> int:
    return int(contains_negation(premise) or contains_negation(hypothesis))


def encode_text_batch(encoder: AutoModel, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = batch["token_type_ids"].to(device)

    outputs = encoder(**inputs)
    sentence_embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    return F.normalize(sentence_embeddings, p=2, dim=1)


def embed_sentences(
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    sentences: Sequence[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoder.eval()
    with torch.no_grad():
        encoded = tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = encoder(**encoded)
        embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)


class MultiNLIDataset(Dataset):
    """将 MultiNLI jsonl 数据封装为可用于 DataLoader 的数据集。"""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        premise_template: str,
        hypothesis_template: str,
        encoding: str,
    ) -> None:
        if not os.path.exists(jsonl_path):
            raise ValueError(f"{jsonl_path} 不存在，请确认 MultiNLI 数据已放置在指定目录。")

        premises: List[str] = []
        hypotheses: List[str] = []
        labels: List[int] = []
        confounders: List[int] = []

        with open(jsonl_path, "r", encoding=encoding) as infile:
            for line in infile:
                if not line.strip():
                    continue
                example = json.loads(line)
                label = str(example.get("gold_label", "")).strip()
                if label not in LABEL2ID:
                    continue

                premise = example.get("sentence1") or example.get("premise")
                hypothesis = example.get("sentence2") or example.get("hypothesis")
                if premise is None or hypothesis is None:
                    continue

                formatted_premise = premise_template.format(premise=str(premise))
                formatted_hypothesis = hypothesis_template.format(hypothesis=str(hypothesis))

                premises.append(formatted_premise)
                hypotheses.append(formatted_hypothesis)
                labels.append(LABEL2ID[label])
                confounders.append(detect_negation(str(premise), str(hypothesis)))

        if not premises:
            raise ValueError(f"{jsonl_path} 中未找到任何有效样本，请检查文件格式。")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = np.array(labels, dtype=int)
        self.confounders = np.array(confounders, dtype=int)
        self.group_ids = self.labels * 2 + self.confounders

    def __len__(self) -> int:
        return len(self.premises)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "confounder": torch.tensor(self.confounders[idx], dtype=torch.long),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item


class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def inference_a(embeddings: torch.Tensor, confounder_embeddings: torch.Tensor) -> torch.Tensor:
    logits = torch.mm(embeddings, confounder_embeddings.t())
    probs = logits.softmax(dim=1)
    _, prediction = torch.max(probs, dim=1)
    return prediction


def supervised_inference_a(embeddings: torch.Tensor, probe: Optional[LinearProbe]) -> torch.Tensor:
    if probe is None:
        raise ValueError("partial_a=True 时必须提供训练好的混杂变量分类器。")
    probe.eval()
    with torch.no_grad():
        logits = probe(embeddings)
        _, prediction = torch.max(logits, dim=1)
    return prediction


def get_confounder_description(confounder_id: int) -> str:
    return "with_negation" if confounder_id == 1 else "without_negation"


def default_confounder_prompts() -> List[str]:
    return [
        "a premise-hypothesis pair without negation words",
        "a premise-hypothesis pair containing negation words",
    ]


def compute_scale(
    encoder: AutoModel,
    confounder_embeddings: torch.Tensor,
    training_loader: DataLoader,
    device: torch.device,
    use_true_confounder: bool,
    partial_a: bool,
    probe: Optional[LinearProbe] = None,
    confounder_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """
    计算每个混杂变量方向的尺度（scale）值。

    该函数是 TIE 方法的核心步骤之一，用于估计混杂变量方向在特征空间中的投影强度。
    """
    encoder.eval()
    num_confounders = confounder_embeddings.size(0)
    normalized_confounders = F.normalize(confounder_embeddings, p=2, dim=1)
    scale_values: List[List[float]] = [[] for _ in range(num_confounders)]

    for batch in tqdm(training_loader, desc="Computing Scale"):
        with torch.no_grad():
            embeddings = encode_text_batch(encoder, batch, device)

            if use_true_confounder:
                sensitive = batch["confounder"].to(device)
            else:
                if partial_a:
                    sensitive = supervised_inference_a(embeddings, probe)
                else:
                    sensitive = inference_a(embeddings, normalized_confounders).long()

            for confounder_id in range(num_confounders):
                mask = sensitive == confounder_id
                if mask.any():
                    confounder_embeddings_local = F.normalize(embeddings[mask], p=2, dim=1)
                    inner = torch.mm(
                        confounder_embeddings_local,
                        normalized_confounders[confounder_id].unsqueeze(1),
                    ).squeeze(1)
                    scale_values[confounder_id].extend(inner.detach().cpu().tolist())

    mean_scales = [float(np.mean(values)) if values else 0.0 for values in scale_values]
    for idx, value in enumerate(mean_scales):
        name = confounder_names[idx] if confounder_names and idx < len(confounder_names) else f"confounder_{idx}"
        print(f"{name} scale: {value:.4f}")

    return torch.tensor(mean_scales, device=device)


def debias_by_confounder(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    confounder_embeddings: torch.Tensor,
    scales: torch.Tensor,
    mode: str = "mean",
    strength: float = 1.0,
) -> torch.Tensor:
    """
    mode:
      - 'mean': x' = x - λ * m_k * c
      - 'projection': x' = x - λ * (x·c) * c
      - 'centered_projection': x' = x - λ * ((x·c) - m_k) * c
    """
    confounder_embeddings = F.normalize(confounder_embeddings, p=2, dim=1)
    adjusted = embeddings.clone()

    for confounder_id in range(confounder_embeddings.size(0)):
        mask = sensitive == confounder_id
        if not mask.any():
            continue
        base = adjusted[mask]
        c_vec = confounder_embeddings[confounder_id].unsqueeze(0)

        if mode == "projection":
            proj = torch.mm(base, c_vec.t()).squeeze(1)
            remove = strength * proj.unsqueeze(1) * c_vec
        elif mode == "centered_projection":
            proj = torch.mm(base, c_vec.t()).squeeze(1)
            centered = (proj - scales[confounder_id]).unsqueeze(1)
            remove = strength * centered * c_vec
        elif mode == "mean":
            remove = strength * scales[confounder_id] * c_vec
        else:
            raise ValueError(f"Unsupported debias_mode: {mode}")

        debiased = base - remove
        adjusted[mask] = F.normalize(debiased, p=2, dim=1)

    return adjusted


def collect_debiased_embeddings(
    encoder: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    confounder_embeddings: torch.Tensor,
    scales: torch.Tensor,
    use_true_confounder: bool,
    partial_a: bool,
    probe: Optional[LinearProbe] = None,
    description: str = "Extracting debiased features",
    debias_mode: str = "mean",
    debias_strength: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
        features: 去偏后的文本特征，形状为 (N, hidden_size)
        labels: 样本标签
        confounders: 样本真实混杂变量（0=无否定, 1=包含否定）
    """
    encoder.eval()
    if probe is not None:
        probe.eval()

    confounder_embeddings = confounder_embeddings.to(device)
    scales = scales.to(device)

    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_confounders: List[torch.Tensor] = []

    for batch in tqdm(dataloader, desc=description):
        labels = batch["label"].to(device)
        confounders_true = batch["confounder"].to(device)

        embeddings = encode_text_batch(encoder, batch, device)

        if use_true_confounder:
            sensitive = confounders_true
        else:
            if partial_a:
                sensitive = supervised_inference_a(embeddings, probe)
            else:
                sensitive = inference_a(embeddings, confounder_embeddings).long()

        adjusted = debias_by_confounder(
            embeddings,
            sensitive,
            confounder_embeddings,
            scales,
            mode=debias_mode,
            strength=debias_strength,
        )

        all_features.append(adjusted.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_confounders.append(confounders_true.detach().cpu())

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    confounder_tensor = torch.cat(all_confounders, dim=0)
    return features_tensor, labels_tensor, confounder_tensor


class GroupBalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        labels: torch.Tensor,
        confounders: torch.Tensor,
        batch_size: int,
        num_labels: int,
        num_confounders: int,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        self.labels = labels
        self.confounders = confounders
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_confounders = num_confounders
        self.num_groups = num_labels * num_confounders

        group_ids_full = labels * num_confounders + confounders
        self.group_indices: List[List[int]] = []
        for g in range(self.num_groups):
            idxs = torch.where(group_ids_full == g)[0].tolist()
            self.group_indices.append(idxs)

        self.has_empty = any(len(v) == 0 for v in self.group_indices)

        base_steps = int(math.ceil(len(labels) / float(batch_size)))
        self.steps = steps_per_epoch or base_steps

        self.per_group = max(self.batch_size // self.num_groups, 1)
        self.remainder = max(self.batch_size - self.per_group * self.num_groups, 0)

    def __iter__(self):
        rng = random.Random()
        for _ in range(self.steps):
            batch: List[int] = []
            for g in range(self.num_groups):
                src = self.group_indices[g]
                if len(src) == 0:
                    continue
                take = [rng.choice(src) for _ in range(self.per_group)]
                batch.extend(take)
            for g in rng.sample(list(range(self.num_groups)), k=self.remainder):
                src = self.group_indices[g]
                if len(src) == 0:
                    continue
                batch.append(rng.choice(src))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.steps


def train_linear_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    confounders: torch.Tensor,
    device: torch.device,
    num_labels: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    num_confounders: int = 2,
    dro_eta: float = 0.05,
    dro_gamma: float = 0.0,
    sampler_type: str = "balanced",
    dro_ema_alpha: float = 0.2,
    warmup_epochs: int = 3,
) -> LinearProbe:
    """
    在固定的去偏特征上训练线性分类器头，结合组级 DRO 以提高最差组性能。
    """
    dataset = TensorDataset(features, labels, confounders)

    num_groups = num_labels * num_confounders
    group_ids_full = labels * num_confounders + confounders
    group_counts = torch.bincount(group_ids_full, minlength=num_groups).float()

    print("[LinearHead] 组样本数:", {int(i): int(c) for i, c in enumerate(group_counts)})

    if sampler_type == "balanced":
        gb_sampler = GroupBalancedBatchSampler(labels, confounders, batch_size, num_labels, num_confounders)
        if not gb_sampler.has_empty:
            loader = DataLoader(dataset, batch_sampler=gb_sampler, num_workers=0, drop_last=False)
        else:
            print("[LinearHead] ⚠️ 发现空组，回退为 WeightedRandomSampler。")
            sampler_type = "weighted"
    if sampler_type == "weighted":
        inv_group_counts = torch.where(group_counts > 0, 1.0 / group_counts, torch.zeros_like(group_counts))
        sample_weights = inv_group_counts[group_ids_full]
        sampler = WeightedRandomSampler(sample_weights.tolist(), len(sample_weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)

    classifier = LinearProbe(features.size(1), num_labels).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    q = torch.ones(num_groups, device=device) / num_groups
    ema_group_losses = torch.zeros(num_groups, device=device)

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for batch_features, batch_labels, batch_confounders in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_confounders = batch_confounders.to(device)

            optimizer.zero_grad()
            logits = classifier(batch_features)
            per_sample_loss = criterion(logits, batch_labels)

            group_ids = batch_labels * num_confounders + batch_confounders
            group_losses = torch.zeros(num_groups, device=device)
            for g in range(num_groups):
                mask = group_ids == g
                if mask.any():
                    group_losses[g] = per_sample_loss[mask].mean()

            if epoch < warmup_epochs:
                weighted_loss = group_losses.mean()
            else:
                if dro_ema_alpha > 0:
                    ema_group_losses = torch.where(
                        group_losses > 0,
                        (1 - dro_ema_alpha) * ema_group_losses + dro_ema_alpha * group_losses,
                        ema_group_losses,
                    )
                    losses_for_q = ema_group_losses.detach()
                else:
                    losses_for_q = group_losses.detach()

                q = q * torch.exp(dro_eta * losses_for_q)
                q = q / q.sum().clamp(min=1e-12)

                weighted_loss = (q * group_losses).sum()
                if dro_gamma > 0:
                    weighted_loss += dro_gamma * q.pow(2).sum()

            weighted_loss.backward()
            optimizer.step()

            total_loss += float(weighted_loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[LinearHead] Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - sampler={sampler_type}")

    classifier.eval()
    return classifier


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def run_epoch(
    encoder: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    use_true_confounder: bool,
    partial_a: bool,
    confounder_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    scales: torch.Tensor,
    probe: Optional[LinearProbe] = None,
    classifier: Optional[nn.Module] = None,
    confounder_names: Optional[Sequence[str]] = None,
    label_texts: Optional[Sequence[str]] = None,
    debias_mode: str = "mean",
    debias_strength: float = 1.0,
) -> None:
    encoder.eval()
    if probe is not None:
        probe.eval()
    if classifier is not None:
        classifier.eval()

    normalized_confounders = F.normalize(confounder_embeddings, p=2, dim=1).to(device)
    normalized_labels = F.normalize(label_embeddings, p=2, dim=1).to(device)
    scales = scales.to(device)

    num_confounders = normalized_confounders.size(0)
    num_labels = normalized_labels.size(0)
    confounder_display = list(confounder_names) if confounder_names else [f"confounder_{i}" for i in range(num_confounders)]
    label_names = list(label_texts) if label_texts else [f"label_{i}" for i in range(num_labels)]

    predictions: List[int] = []
    labels_all: List[int] = []
    confounders_all: List[int] = []

    correct_combo = np.zeros((num_labels, num_confounders), dtype=float)
    total_combo = np.zeros((num_labels, num_confounders), dtype=float)

    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            labels = batch["label"].to(device)
            confounders_true = batch["confounder"].to(device)
            embeddings = encode_text_batch(encoder, batch, device)

            if use_true_confounder:
                sensitive = confounders_true
            else:
                if partial_a:
                    sensitive = supervised_inference_a(embeddings, probe)
                else:
                    sensitive = inference_a(embeddings, normalized_confounders).long()

            adjusted = debias_by_confounder(
                embeddings,
                sensitive,
                normalized_confounders,
                scales,
                mode=debias_mode,
                strength=debias_strength,
            )

            if classifier is not None:
                logits = classifier(adjusted)
            else:
                logits = torch.mm(adjusted, normalized_labels.t())

            probs = logits.softmax(dim=1)
            _, pred = torch.max(probs, dim=1)

            labels_cpu = labels.detach().cpu()
            confounders_cpu = confounders_true.detach().cpu()
            pred_cpu = pred.detach().cpu()

            predictions.extend(pred_cpu.numpy().tolist())
            labels_all.extend(labels_cpu.numpy().tolist())
            confounders_all.extend(confounders_cpu.numpy().tolist())

            for label_idx in range(num_labels):
                for confounder_idx in range(num_confounders):
                    mask = (labels_cpu == label_idx) & (confounders_cpu == confounder_idx)
                    if mask.any():
                        matches = (pred_cpu[mask] == label_idx).sum().item()
                        total = mask.sum().item()
                        correct_combo[label_idx, confounder_idx] += matches
                        total_combo[label_idx, confounder_idx] += total

    overall_acc = accuracy_score(labels_all, predictions)
    print(f"Overall accuracy: {overall_acc:.4f}")

    print("\n" + "=" * 80)
    print("按标签与混杂变量划分的分类准确率：")
    print("=" * 80)
    for label_idx in range(num_labels):
        label_name = label_names[label_idx]
        for confounder_idx in range(num_confounders):
            acc = safe_div(correct_combo[label_idx, confounder_idx], total_combo[label_idx, confounder_idx])
            conf_name = confounder_display[confounder_idx] if confounder_idx < len(confounder_display) else f"conf{confounder_idx}"
            print(
                f"Accuracy for label={label_name}, confounder={conf_name}: "
                f"{acc:.4f} (样本数: {int(total_combo[label_idx, confounder_idx])})"
            )

    confounder_metrics = {}
    predictions_np = np.array(predictions)
    labels_np = np.array(labels_all)
    confounders_np = np.array(confounders_all)
    for confounder_idx in range(num_confounders):
        mask = confounders_np == confounder_idx
        if not mask.any():
            continue
        conf_acc = accuracy_score(labels_np[mask], predictions_np[mask])
        conf_name = confounder_display[confounder_idx] if confounder_idx < len(confounder_display) else f"conf{confounder_idx}"
        confounder_metrics[conf_name] = conf_acc
        print(f"{conf_name} accuracy: {conf_acc:.4f}")

    if confounder_metrics:
        values = list(confounder_metrics.values())
        print(f"Confounder accuracy gap (max-min): {max(values) - min(values):.4f}")

    if num_confounders == 2:
        mask0 = confounders_np == 0
        mask1 = confounders_np == 1
        if mask0.any() and mask1.any():
            cm0 = confusion_matrix(labels_np[mask0], predictions_np[mask0], labels=list(range(num_labels)))
            cm1 = confusion_matrix(labels_np[mask1], predictions_np[mask1], labels=list(range(num_labels)))
            print("\n公平性指标：")
            for label_idx in range(num_labels):
                denom0 = cm0[label_idx].sum()
                denom1 = cm1[label_idx].sum()
                if denom0 == 0 or denom1 == 0:
                    continue
                tpr0 = safe_div(cm0[label_idx, label_idx], denom0)
                tpr1 = safe_div(cm1[label_idx, label_idx], denom1)
                print(
                    f"  标签 {label_names[label_idx]} 的 TPR 差异: {abs(tpr0 - tpr1):.4f} "
                    f"(无否定: {tpr0:.4f}, 否定: {tpr1:.4f})"
                )


def load_probe(
    path: Optional[str],
    hidden_size: int,
    num_classes: int,
    device: torch.device,
) -> Optional[LinearProbe]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise ValueError(f"找不到混杂变量分类器权重文件：{path}")

    probe = LinearProbe(hidden_size, num_classes)
    state_dict = torch.load(path, map_location=device)
    probe.load_state_dict(state_dict)
    probe.to(device)
    return probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIE 在 MultiNLI 数据集上的实现")
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default=os.path.join("datasets", "multinli", "multinli_1.0_train.jsonl"),
        help="MultiNLI 训练集 jsonl 路径",
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default=os.path.join("datasets", "multinli", "multinli_1.0_dev_matched.jsonl"),
        help="MultiNLI 验证/测试集 jsonl 路径",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="e5-base-v2",
        help="预训练文本编码器名称或本地路径",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_encoding", type=str, default="utf-8", help="读取 jsonl 时使用的编码")
    parser.add_argument(
        "--use_true_confounder",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否使用真实的混杂变量标签（true 表示 TIE，false 表示 TIE*），默认为 true",
    )
    parser.add_argument("--partial_a", action="store_true", help="若指定则使用线性探针预测部分混杂变量")
    parser.add_argument("--confounder_head_path", type=str, default=None, help="混杂变量线性探针权重路径，仅在 partial_a=True 时使用")
    parser.add_argument(
        "--label_prompts",
        nargs="+",
        default=[
            "a hypothesis that contradicts the premise",
            "a hypothesis that is neutral with respect to the premise",
            "a hypothesis that is entailed by the premise",
        ],
        help="描述 MultiNLI 标签的提示文本，顺序需与标签映射一致：contradiction、neutral、entailment",
    )
    parser.add_argument(
        "--confounder_prompts",
        nargs="*",
        default=None,
        help="描述混杂变量（是否包含否定词）的提示文本，需提供两个，顺序为 [无否定, 有否定]。未提供时自动生成。",
    )
    parser.add_argument(
        "--premise_template",
        type=str,
        default="{premise}",
        help="Premise 文本模板（需包含 {premise} 占位符）",
    )
    parser.add_argument(
        "--hypothesis_template",
        type=str,
        default="{hypothesis}",
        help="Hypothesis 文本模板（需包含 {hypothesis} 占位符）",
    )
    parser.add_argument(
        "--train_linear_head",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否在去偏特征上训练线性分类头（默认：true）",
    )
    parser.add_argument("--head_epochs", type=int, default=50, help="线性分类头训练轮数")
    parser.add_argument("--head_learning_rate", type=float, default=5e-3, help="线性分类头学习率")
    parser.add_argument("--head_weight_decay", type=float, default=1e-4, help="线性分类头权重衰减系数")
    parser.add_argument("--head_batch_size", type=int, default=256, help="线性分类头训练的 batch 大小")
    parser.add_argument("--head_dro_eta", type=float, default=0.1, help="组级 DRO 的步长参数（eta）")
    parser.add_argument("--head_dro_gamma", type=float, default=0.1, help="DRO 权重正则化系数，稳定权重分布")
    parser.add_argument(
        "--head_sampler",
        type=str,
        default="balanced",
        choices=["balanced", "weighted"],
        help="线性头训练采样器：balanced=组均衡 batch；weighted=按逆频率加权采样",
    )
    parser.add_argument("--head_dro_ema", type=float, default=0.2, help="DRO 组损失 EMA 平滑系数 (0 关闭)")
    parser.add_argument("--head_warmup_epochs", type=int, default=10, help="DRO 权重 q 的预热 epoch 数")
    parser.add_argument(
        "--debias_mode",
        type=str,
        default="mean",
        choices=["mean", "projection", "centered_projection"],
        help="去偏方式：mean(推荐)/projection/centered_projection",
    )
    parser.add_argument(
        "--debias_strength",
        type=float,
        default=0.75,
        help="去偏强度 λ（0~1 常用，>1 更激进，谨慎）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    use_true_confounder = args.use_true_confounder.lower() == "true"
    train_linear_head_flag = args.train_linear_head.lower() == "true"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"使用真实混杂变量标签: {use_true_confounder} ({'TIE' if use_true_confounder else 'TIE*'})")
    print(f"训练去偏线性分类头: {train_linear_head_flag}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_name)
    encoder = encoder.to(device)

    train_dataset = MultiNLIDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
        premise_template=args.premise_template,
        hypothesis_template=args.hypothesis_template,
        encoding=args.json_encoding,
    )
    eval_dataset = MultiNLIDataset(
        jsonl_path=args.eval_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
        premise_template=args.premise_template,
        hypothesis_template=args.hypothesis_template,
        encoding=args.json_encoding,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    if args.confounder_prompts:
        if len(args.confounder_prompts) != 2:
            raise ValueError(
                f"对于否定词混杂变量，需要提供两个提示文本，但接收到 {len(args.confounder_prompts)} 个。"
            )
        confounder_prompts = list(args.confounder_prompts)
    else:
        confounder_prompts = default_confounder_prompts()

    confounder_embeddings = embed_sentences(
        encoder=encoder,
        tokenizer=tokenizer,
        sentences=confounder_prompts,
        device=device,
        max_length=args.max_length,
    )
    print(f"混杂变量（否定词）嵌入向量形状: {confounder_embeddings.shape} (应为 (2, hidden_size))")

    if len(args.label_prompts) != len(LABEL2ID):
        raise ValueError(
            f"label_prompts 数量（{len(args.label_prompts)}）需要与标签种类数（{len(LABEL2ID)}）一致。"
        )

    label_embeddings = embed_sentences(
        encoder=encoder,
        tokenizer=tokenizer,
        sentences=args.label_prompts,
        device=device,
        max_length=args.max_length,
    )

    probe = None
    if args.partial_a:
        hidden_size = encoder.config.hidden_size
        num_confounders = confounder_embeddings.size(0)
        probe = load_probe(args.confounder_head_path, hidden_size, num_confounders, device)

    confounder_names = [get_confounder_description(idx) for idx in range(confounder_embeddings.size(0))]
    label_names = [ID2LABEL[idx] for idx in range(len(ID2LABEL))]

    print("Estimating negation-specific scales on training data...")
    scales = compute_scale(
        encoder=encoder,
        confounder_embeddings=confounder_embeddings,
        training_loader=train_loader,
        device=device,
        use_true_confounder=use_true_confounder,
        partial_a=args.partial_a,
        probe=probe,
        confounder_names=confounder_names,
    )

    classifier: Optional[LinearProbe] = None
    if train_linear_head_flag:
        print("Collecting debiased training features for linear head...")
        train_features, train_labels_tensor, train_confounders_tensor = collect_debiased_embeddings(
            encoder=encoder,
            dataloader=train_loader,
            device=device,
            confounder_embeddings=confounder_embeddings,
            scales=scales,
            use_true_confounder=use_true_confounder,
            partial_a=args.partial_a,
            probe=probe,
            description="Collecting train features",
            debias_mode=args.debias_mode,
            debias_strength=args.debias_strength,
        )
        classifier = train_linear_head(
            features=train_features,
            labels=train_labels_tensor,
            confounders=train_confounders_tensor,
            device=device,
            num_labels=len(label_names),
            epochs=args.head_epochs,
            lr=args.head_learning_rate,
            weight_decay=args.head_weight_decay,
            batch_size=args.head_batch_size,
            num_confounders=confounder_embeddings.size(0),
            dro_eta=args.head_dro_eta,
            dro_gamma=args.head_dro_gamma,
            sampler_type=args.head_sampler,
            dro_ema_alpha=args.head_dro_ema,
            warmup_epochs=args.head_warmup_epochs,
        )
        with torch.no_grad():
            train_logits = classifier(train_features.to(device))
            train_pred = train_logits.argmax(dim=1).cpu()
            train_acc = (train_pred == train_labels_tensor).float().mean().item()
        print(f"线性分类头训练集准确率: {train_acc:.4f}")
        del train_features, train_labels_tensor, train_confounders_tensor, train_logits, train_pred

    print("Starting evaluation...")
    print("=" * 80)
    print("数据集信息：")
    print(f"  - 标签映射：{ID2LABEL}")
    print(f"  - 混杂变量：是否包含否定词（0=without_negation, 1=with_negation）")
    print("  - 六组组合：")
    for label_idx, label_name in enumerate(label_names):
        for conf_idx, conf_name in enumerate(confounder_names):
            group_id = label_idx * len(confounder_names) + conf_idx
            print(f"    group_id={group_id}: label={label_name}, confounder={conf_name}")
    print("=" * 80)

    run_epoch(
        encoder=encoder,
        dataloader=eval_loader,
        device=device,
        use_true_confounder=use_true_confounder,
        partial_a=args.partial_a,
        confounder_embeddings=confounder_embeddings,
        label_embeddings=label_embeddings,
        probe=probe,
        scales=scales,
        classifier=classifier,
        confounder_names=confounder_names,
        label_texts=label_names,
        debias_mode=args.debias_mode,
        debias_strength=args.debias_strength,
    )


if __name__ == "__main__":
    main()


