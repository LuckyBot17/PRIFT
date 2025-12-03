import argparse
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler, Sampler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

torch.set_num_threads(5)
torch.set_num_interop_threads(5)

# 函数：固定所有随机种子，保证实验可复现
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 函数：对token 级输出做 mean pooling，得到句向量表示
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


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


# 数据集封装：从 CSV 读取 Yelp 文本、标签、group_id，并在 __getitem__ 中完成分词编码
class YelpAuthorStyleDataset(Dataset):
    """将 Yelp Author Style CSV 数据封装为可用于 DataLoader 的数据集。"""

    def __init__(
            self,
            csv_path: str,
            tokenizer: AutoTokenizer,
            group2id: Dict[str, int],
            max_length: int,
            text_template: str,
            encoding: str,
    ) -> None:
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} 不存在，请确认 Yelp Author Style 数据已放置在指定目录。")

        dataframe = pd.read_csv(csv_path, encoding=encoding)
        required_columns = {"text", "label", "group_ids"}
        missing = required_columns - set(dataframe.columns)
        if missing:
            raise ValueError(f"CSV 文件缺少必要列：{missing}")

        texts = dataframe["text"].astype(str).tolist()
        labels = dataframe["label"].astype(int).to_numpy()
        group_raw = dataframe["group_ids"].astype(str).tolist()

        try:
            confounders = np.array([group2id[g] for g in group_raw], dtype=int)
        except KeyError as exc:
            raise ValueError(f"发现未在映射中的 group_id：{exc.args[0]}") from exc

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_template = text_template
        self.formatted_texts = [text_template.format(text=txt) for txt in texts]
        self.labels = labels
        self.confounders = confounders

    # 方法：返回样本数量
    def __len__(self) -> int:
        return len(self.formatted_texts)

    # 方法：取第 idx 条样本，并使用 tokenizer 将文本转为模型输入张量
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.formatted_texts[idx],
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


# 线性探针：在固定文本特征上做线性分类，用于预测标签或混杂变量
class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


# 函数：用与混杂变量原型向量（风格嵌入）的相似度进行最近原型推断，得到敏感属性预测
def inference_a(embeddings: torch.Tensor, confounder_embeddings: torch.Tensor) -> torch.Tensor:
    logits = torch.mm(embeddings, confounder_embeddings.t())
    probs = logits.softmax(dim=1)
    _, prediction = torch.max(probs, dim=1)
    return prediction

# 函数：使用已训练的线性探针对敏感属性进行监督推断
def supervised_inference_a(embeddings: torch.Tensor, probe: Optional[LinearProbe]) -> torch.Tensor:
    if probe is None:
        raise ValueError("partial_a=True 时必须提供训练好的混杂变量分类器。")
    probe.eval()
    with torch.no_grad():
        logits = probe(embeddings)
        _, prediction = torch.max(logits, dim=1)
    return prediction

# 函数：对字符串组号做“数值排序（若都是数字）或字面排序”，确保映射顺序稳定
def _numeric_sorted_unique(groups: Sequence[str]) -> List[str]:
    # [MOD] 若全是数字字符串，按数值排序，避免 "10" < "2" 的字面排序陷阱
    if all(g.isdigit() for g in groups):
        return [x for x in sorted(groups, key=lambda z: int(z))]
    return sorted(groups)


# 函数：读取 train/eval CSV 的 group_ids 列并建立从原始 group 到连续 ID 的映射字典
def build_group_mapping(
        train_csv: str,
        eval_csv: Optional[str],
        encoding: str,
) -> Dict[str, int]:
    """
    构建 group_ids 到内部 ID 的映射。

    对于 yelp-author-style 数据集：
    - group_id 0: label=0 (negative), style=Hemingway
    - group_id 1: label=0 (negative), style=Shakespeare
    - group_id 2: label=1 (positive), style=Hemingway
    - group_id 3: label=1 (positive), style=Shakespeare
    """
    paths = [train_csv]
    if eval_csv:
        paths.append(eval_csv)

    frames = []
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"未找到 CSV 文件：{path}")
        frames.append(pd.read_csv(path, encoding=encoding, usecols=["group_ids"]))

    all_groups = pd.concat(frames, axis=0)["group_ids"].astype(str)
    unique_groups = _numeric_sorted_unique(all_groups.unique())  # [MOD]
    if not unique_groups:
        raise ValueError("在数据集中没有发现任何 group_ids。")

    return {group: idx for idx, group in enumerate(unique_groups)}


# 函数：将组 ID 转为更友好的可读描述（用于日志与可视化）
def get_group_description(group_id: int) -> str:
    descriptions = {
        0: "negative-Hemingway",
        1: "negative-Shakespeare",
        2: "positive-Hemingway",
        3: "positive-Shakespeare",
    }
    return descriptions.get(group_id, f"group_{group_id}")

# 函数：在训练集上估计每个风格方向上的平均相似度（scale），用于后续去偏强度的自适应
def compute_scale(
        encoder: AutoModel,
        confounder_embeddings: torch.Tensor,
        training_loader: DataLoader,
        device: torch.device,
        use_true_confounder: bool,
        partial_a: bool,
        probe: Optional[LinearProbe] = None,
) -> torch.Tensor:
    encoder.eval()
    num_styles = confounder_embeddings.size(0)
    normalized_styles = F.normalize(confounder_embeddings, p=2, dim=1)
    scale_values: List[List[float]] = [[] for _ in range(num_styles)]

    for batch in tqdm(training_loader, desc="Computing Scale"):
        with torch.no_grad():
            embeddings = encode_text_batch(encoder, batch, device) #得到样本文本的句向量（已 mean pooling + 归一化）

            if use_true_confounder:
                sensitive = (batch["confounder"].to(device) % 2).long()
            else:
                if partial_a:
                    sensitive = supervised_inference_a(embeddings, probe)
                    sensitive = (sensitive % 2).long()
                else:
                    sensitive = inference_a(embeddings, normalized_styles).long()

            for style_id in range(num_styles):
                mask = sensitive == style_id
                if mask.any():
                    style_embeddings = F.normalize(embeddings[mask], p=2, dim=1)
                    inner = torch.mm(style_embeddings, normalized_styles[style_id].unsqueeze(1)).squeeze(1)
                    scale_values[style_id].extend(inner.detach().cpu().tolist())

    mean_scales = [float(np.mean(values)) if values else 0.0 for values in scale_values]
    style_names = ["Hemingway", "Shakespeare"]
    for idx, value in enumerate(mean_scales):
        print(f"{style_names[idx]} style scale: {value:.4f}")
    return torch.tensor(mean_scales, device=device)



# 函数：按指定模式对嵌入进行去偏（mean/projection/centered_projection）
def debias_by_style(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    style_embeddings: torch.Tensor,
    scales: torch.Tensor,
    mode: str = "mean",
    strength: float = 1.0,
) -> torch.Tensor:
    """
    mode:
      - 'mean': x' = x - λ * m_k * s
      - 'projection': x' = x - λ * (x·s) * s
      - 'centered_projection': x' = x - λ * ((x·s) - m_k) * s
    """
    style_embeddings = F.normalize(style_embeddings, p=2, dim=1)
    adjusted = embeddings.clone()

    for style_id in range(style_embeddings.size(0)):
        mask = (sensitive == style_id)
        if not mask.any():
            continue
        base = adjusted[mask]
        s_vec = style_embeddings[style_id].unsqueeze(0)  # (1, d)

        if mode == "projection":
            proj = torch.mm(base, s_vec.t()).squeeze(1)  # (n,)
            remove = strength * proj.unsqueeze(1) * s_vec
        elif mode == "centered_projection":
            proj = torch.mm(base, s_vec.t()).squeeze(1)  # (n,)
            centered = (proj - scales[style_id]).unsqueeze(1)
            remove = strength * centered * s_vec
        elif mode == "mean":
            remove = strength * scales[style_id] * s_vec  # broadcast
        else:
            raise ValueError(f"Unsupported debias_mode: {mode}")

        debiased = base - remove
        adjusted[mask] = F.normalize(debiased, p=2, dim=1)

    return adjusted


# 函数：遍历数据集，抽取句向量并执行去偏，汇总为训练线性头所需的特征/标签/风格张量
def collect_debiased_embeddings(
        encoder: AutoModel,
        dataloader: DataLoader,
        device: torch.device,
        style_embeddings: torch.Tensor,
        scales: torch.Tensor,
        use_true_confounder: bool,
        partial_a: bool,
        probe: Optional[LinearProbe] = None,
        description: str = "Extracting debiased features",
        debias_mode: str = "mean",
        debias_strength: float = 1.0, #去偏强度
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    模型中负责“提取去偏文本特征”
    返回:
        features: 去偏后的文本特征，形状为 (N, hidden_size)
        labels: 样本标签
        styles: 样本写作风格（0=Hemingway, 1=Shakespeare）
    """
    encoder.eval()
    if probe is not None:
        probe.eval()

    style_embeddings = style_embeddings.to(device)
    scales = scales.to(device)

    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_styles: List[torch.Tensor] = []

    for batch in tqdm(dataloader, desc=description):
        labels = batch["label"].to(device)
        confounders = batch["confounder"].to(device)
        style_real = (confounders % 2).long() #将0,1,2,3转为0,1,0,1

        embeddings = encode_text_batch(encoder, batch, device)

        if use_true_confounder:
            sensitive = style_real
        else:
            if partial_a:
                sensitive = supervised_inference_a(embeddings, probe)
                sensitive = (sensitive % 2).long()
            else:
                sensitive = inference_a(embeddings, style_embeddings).long()


        adjusted = debias_by_style(
            embeddings, sensitive, style_embeddings, scales,
            mode=debias_mode, strength=debias_strength
        )

        all_features.append(adjusted.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_styles.append(style_real.detach().cpu())

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    styles_tensor = torch.cat(all_styles, dim=0)
    return features_tensor, labels_tensor, styles_tensor #features_tensor：去偏后的文本向量；labels_tensor：样本的情感标签；styles_tensor：风格（Hemingway / Shakespeare）



class GroupBalancedBatchSampler(Sampler[List[int]]):
    # 方法：根据(label, style)组合分组采样，近似保证每个 batch 的组别均衡
    def __init__(
        self,
        labels: torch.Tensor,
        styles: torch.Tensor,
        batch_size: int,
        num_labels: int,
        num_styles: int,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        self.labels = labels
        self.styles = styles
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_styles = num_styles
        self.num_groups = num_labels * num_styles

        group_ids_full = labels * num_styles + styles
        self.group_indices: List[List[int]] = []
        for g in range(self.num_groups):
            idxs = torch.where(group_ids_full == g)[0].tolist()
            self.group_indices.append(idxs)


        self.has_empty = any(len(v) == 0 for v in self.group_indices)

        base_steps = int(math.ceil(len(labels) / float(batch_size)))
        self.steps = steps_per_epoch or base_steps

        self.per_group = max(self.batch_size // self.num_groups, 1)
        self.remainder = max(self.batch_size - self.per_group * self.num_groups, 0)

    # 方法：生成一个 epoch 内的若干 batch 索引列表
    def __iter__(self):
        rng = random.Random()
        for _ in range(self.steps):
            batch: List[int] = []
            # 每组采样 per_group 个
            for g in range(self.num_groups):
                src = self.group_indices[g]
                if len(src) == 0:
                    continue
                take = [rng.choice(src) for _ in range(self.per_group)]
                batch.extend(take)
            # 补足 remainder
            for g in rng.sample(list(range(self.num_groups)), k=self.remainder):
                src = self.group_indices[g]
                if len(src) == 0:
                    continue
                batch.append(rng.choice(src))
            rng.shuffle(batch)
            yield batch

    # 方法：返回一个 epoch 中 batch 的数量
    def __len__(self) -> int:
        return self.steps



def train_linear_head(
        features: torch.Tensor,
        labels: torch.Tensor,
        styles: torch.Tensor,
        device: torch.device,
        num_labels: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        num_styles: int = 2,
        dro_eta: float = 0.05,
        dro_gamma: float = 0.0,
        sampler_type: str = "balanced",  # [MOD] 新增
        dro_ema_alpha: float = 0.2,      # [MOD] 新增：DRO 组损失 EMA 权重
        warmup_epochs: int = 3,          # [MOD] 新增：q 的预热 epoch 数
) -> LinearProbe:
    """
    在固定的去偏特征上训练线性分类器头，结合组级 DRO 以提高最差分组性能。
    """
    dataset = TensorDataset(features, labels, styles)

    num_groups = num_labels * num_styles
    group_ids_full = labels * num_styles + styles
    group_counts = torch.bincount(group_ids_full, minlength=num_groups).float()

    # 打印组样本数，便于核查分布与映射是否异常
    print("各组样本数:", {int(i): int(c) for i, c in enumerate(group_counts)})

    # 采样器：优先使用组均衡 batch，遇到空组时回退到加权采样
    if sampler_type == "balanced":
        gb_sampler = GroupBalancedBatchSampler(labels, styles, batch_size, num_labels, num_styles)
        if not gb_sampler.has_empty:
            loader = DataLoader(dataset, batch_sampler=gb_sampler, num_workers=0, drop_last=False)
        else:
            print("[LinearHead] ⚠️ 发现空组，回退为 WeightedRandomSampler。")
            sampler_type = "weighted"  # fallthrough
    if sampler_type == "weighted":
        inv_group_counts = torch.where(group_counts > 0, 1.0 / group_counts, torch.zeros_like(group_counts))
        sample_weights = inv_group_counts[group_ids_full]
        sampler = WeightedRandomSampler(sample_weights.tolist(), len(sample_weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)

    classifier = LinearProbe(features.size(1), num_labels).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    q = torch.ones(num_groups, device=device) / num_groups
    ema_group_losses = torch.zeros(num_groups, device=device)  # [MOD]

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for batch_features, batch_labels, batch_styles in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_styles = batch_styles.to(device)

            optimizer.zero_grad()
            logits = classifier(batch_features)
            per_sample_loss = criterion(logits, batch_labels)

            group_ids = batch_labels * num_styles + batch_styles
            group_losses = torch.zeros(num_groups, device=device)
            for g in range(num_groups):
                mask = group_ids == g
                if mask.any():
                    group_losses[g] = per_sample_loss[mask].mean()

            #  warmup：前若干 epoch 固定 q 为均匀，避免早期剧烈震荡
            if epoch < warmup_epochs:
                weighted_loss = (group_losses.mean())  # 等权平均
            else:
                # 用 EMA 平滑组损失，再更新 q，抑制抖动
                if dro_ema_alpha > 0:
                    ema_group_losses = torch.where(
                        group_losses > 0,
                        (1 - dro_ema_alpha) * ema_group_losses + dro_ema_alpha * group_losses,
                        ema_group_losses
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


# 函数：安全除法，分母为 0 时返回 0
def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# 函数：在验证/测试集上评估模型，输出总体准确率、组别准确率与公平性指标
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
        id2confounder: Optional[Dict[int, str]] = None,
        label_texts: Optional[Sequence[str]] = None,
        debias_mode: str = "mean",        # [MOD] 新增
        debias_strength: float = 1.0,     # [MOD] 新增
) -> None:
    encoder.eval()
    if probe is not None:
        probe.eval()
    if classifier is not None:
        classifier.eval()

    style_embeddings = F.normalize(confounder_embeddings, p=2, dim=1).to(device)
    scales = scales.to(device)
    normalized_label_embeddings = F.normalize(label_embeddings, p=2, dim=1).to(device)

    num_styles = style_embeddings.size(0)
    num_labels = normalized_label_embeddings.size(0)
    style_names = ["Hemingway", "Shakespeare"]
    label_names = list(label_texts) if label_texts else [f"label_{i}" for i in range(num_labels)]

    predictions: List[int] = []
    labels_all: List[int] = []
    styles_all: List[int] = []
    feature_style0: List[np.ndarray] = []
    feature_style1: List[np.ndarray] = []
    style0_pred: List[int] = []
    style0_gt: List[int] = []
    style1_pred: List[int] = []
    style1_gt: List[int] = []

    correct_combo = np.zeros((num_labels, num_styles), dtype=float)
    total_combo = np.zeros((num_labels, num_styles), dtype=float)

    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            labels = batch["label"].to(device)
            confounders = batch["confounder"].to(device)
            style_real = (confounders % 2).long()

            embeddings = encode_text_batch(encoder, batch, device)

            if use_true_confounder:
                sensitive = style_real
            else:
                if partial_a:
                    sensitive = supervised_inference_a(embeddings, probe)
                    sensitive = (sensitive % 2).long()
                else:
                    sensitive = inference_a(embeddings, style_embeddings).long()


            adjusted = debias_by_style(
                embeddings, sensitive, style_embeddings, scales,
                mode=debias_mode, strength=debias_strength
            )

            if classifier is not None:
                logits = classifier(adjusted)
            else:
                logits = torch.mm(adjusted, normalized_label_embeddings.t())

            probs = logits.softmax(dim=1)
            _, pred = torch.max(probs, dim=1)

            labels_cpu = labels.detach().cpu()
            style_real_cpu = style_real.detach().cpu()
            pred_cpu = pred.detach().cpu()

            predictions.extend(pred_cpu.numpy().tolist())
            labels_all.extend(labels_cpu.numpy().tolist())
            styles_all.extend(style_real_cpu.numpy().tolist())

            for label_idx in range(num_labels):
                for style_idx in range(num_styles):
                    mask = (labels_cpu == label_idx) & (style_real_cpu == style_idx)
                    if mask.any():
                        matches = (pred_cpu[mask] == label_idx).sum().item()
                        total = mask.sum().item()
                        correct_combo[label_idx, style_idx] += matches
                        total_combo[label_idx, style_idx] += total

            for idx in range(len(pred_cpu)):
                if style_real_cpu[idx] == 0:
                    style0_pred.append(int(pred_cpu[idx]))
                    style0_gt.append(int(labels_cpu[idx]))
                else:
                    style1_pred.append(int(pred_cpu[idx]))
                    style1_gt.append(int(labels_cpu[idx]))

    overall_acc = accuracy_score(labels_all, predictions)
    print(f"Overall accuracy: {overall_acc:.4f}")

    print("\n" + "=" * 80)
    print("四种情况的分类准确率：")
    print("=" * 80)
    for label_idx in range(num_labels):
        label_name = label_names[label_idx]
        for style_idx in range(num_styles):
            acc = safe_div(correct_combo[label_idx, style_idx], total_combo[label_idx, style_idx])
            print(
                f"Accuracy for label={label_name}, style={style_names[style_idx]}: "
                f"{acc:.4f} (样本数: {int(total_combo[label_idx, style_idx])})"
            )

    if feature_style0:
        _ = torch.from_numpy(np.mean(np.stack(feature_style0), axis=0))
    if feature_style1:
        _ = torch.from_numpy(np.mean(np.stack(feature_style1), axis=0))

    if style0_gt and style1_gt:
        cm_style0 = confusion_matrix(style0_gt, style0_pred, labels=list(range(num_labels)))
        cm_style1 = confusion_matrix(style1_gt, style1_pred, labels=list(range(num_labels)))

        dp_style0 = safe_div(cm_style0[:, 1].sum(), cm_style0.sum())
        dp_style1 = safe_div(cm_style1[:, 1].sum(), cm_style1.sum())

        tpr_style0 = safe_div(cm_style0[1, 1], cm_style0[1].sum())
        tpr_style1 = safe_div(cm_style1[1, 1], cm_style1[1].sum())

        fpr_style0 = safe_div(cm_style0[0, 1], cm_style0[0].sum())
        fpr_style1 = safe_div(cm_style1[0, 1], cm_style1[0].sum())

        print("\n公平性指标：")
        print(f"  Demographic parity gap: {abs(dp_style0 - dp_style1):.4f}")
        print(f"  Equal opportunity gap (TPR diff): {abs(tpr_style0 - tpr_style1):.4f}")
        print(f"  Equalized odds gap: {0.5 * (abs(fpr_style0 - fpr_style1) + abs(tpr_style0 - tpr_style1)):.4f}")


# 函数：从磁盘加载已训练的线性探针权重，并实例化为可用模型
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


# 函数：解析命令行参数，汇总本脚本所需的所有可配置项
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIE 在 Yelp Author Style 文本数据集上的实现")
    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join("datasets", "yelp-author-style", "train.csv"),
        help="Yelp Author Style 训练集 CSV 路径",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=os.path.join("datasets", "yelp-author-style", "test.csv"),
        help="Yelp Author Style 验证/测试集 CSV 路径",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="e5-base-v2",
        help="预训练文本编码器名称或本地路径",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv_encoding", type=str, default="latin1", help="读取 CSV 时使用的编码")
    parser.add_argument(
        "--use_true_confounder",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否使用真实的混杂变量标签（true 表示 TIE，false 表示 TIE*），默认为 true"
    )
    parser.add_argument("--partial_a", action="store_true", help="若指定则使用线性探针预测部分混杂变量")
    parser.add_argument("--confounder_head_path", type=str, default=None,
                        help="混杂变量线性探针权重路径，仅在 partial_a=True 时使用")
    parser.add_argument(
        "--train_linear_head",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否训练去偏后的线性分类头（默认：true）",
    )
    parser.add_argument("--head_epochs", type=int, default=25, help="线性分类头训练轮数")
    parser.add_argument("--head_learning_rate", type=float, default=5e-3, help="线性分类头学习率")
    parser.add_argument("--head_weight_decay", type=float, default=1e-4, help="线性分类头权重衰减系数")
    parser.add_argument("--head_batch_size", type=int, default=256, help="线性分类头训练的 batch 大小")
    parser.add_argument("--head_dro_eta", type=float, default=0.05, help="组级 DRO 的步长参数（eta）")
    parser.add_argument("--head_dro_gamma", type=float, default=0.1, help="DRO 权重正则化系数，稳定权重分布")  # [MOD] 调高默认
    # [MOD] 新增的训练稳定性参数
    parser.add_argument("--head_sampler", type=str, default="balanced", choices=["balanced", "weighted"],
                        help="线性头训练采样器：balanced=组均衡 batch；weighted=按逆频率加权采样")
    parser.add_argument("--head_dro_ema", type=float, default=0.2, help="DRO 组损失 EMA 平滑系数 (0 关闭)")
    parser.add_argument("--head_warmup_epochs", type=int, default=3, help="DRO 权重 q 的预热 epoch 数")

    parser.add_argument(
        "--label_prompts",
        nargs="+",
        default=[
            "1",
            "哈哈哈",
        ],
        help="描述类别的提示文本，顺序需与 label 对应",
    )
    # parser.add_argument(
    #     "--label_prompts",
    #     nargs="+",
    #     default=[
    #         "The following is a Yelp review that conveys a negative sentiment",
    #         "The following is a Yelp review that conveys a positive sentiment",
    #     ],
    #     help="描述类别的提示文本，顺序需与 label 对应",
    # )
    parser.add_argument(
        "--group_prompts",
        nargs="*",
        default=[
            "a yelp review written in Hemingway style",
            "a yelp review written in Shakespeare style",
        ],
        help="描述干扰变量 group 的提示文本，顺序需与 group 映射一致。",
    )
    # parser.add_argument(
    #     "--group_prompts",
    #     nargs="*",
    #     default=[
    #         "Short, direct, and concise sentences",
    #         "Old English, poetic, and complex sentences",
    #     ],
    #     help="描述干扰变量 group 的提示文本，顺序需与 group 映射一致。",
    # )
    parser.add_argument(
        "--text_template",
        type=str,
        default="{text}",
        help="构造输入文本的模板（需包含 {text} 占位符）",
    )
    # [MOD] 新增去偏策略与强度
    parser.add_argument("--debias_mode", type=str, default="mean",
                        choices=["mean", "projection", "centered_projection"],
                        help="去偏方式：mean/projection/centered_projection")
    parser.add_argument("--debias_strength", type=float, default=1.0,
                        help="去偏强度 λ （0~1 常用，>1 更激进，谨慎）")
    return parser.parse_args()


# 主函数：构建数据与模型、估计 scale、提取去偏特征、训练线性头并最终评估
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    use_true_confounder = args.use_true_confounder.lower()
    train_linear_head_flag = args.train_linear_head.lower()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"使用真实混杂变量标签: {use_true_confounder} ({'TIE' if use_true_confounder else 'TIE*'})")
    print(f"训练去偏线性分类头: {train_linear_head_flag}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_name)
    encoder = encoder.to(device)

    group2id = build_group_mapping(args.train_csv, args.eval_csv, args.csv_encoding)
    id2group_raw = {idx: group for group, idx in group2id.items()}

    train_dataset = YelpAuthorStyleDataset(
        csv_path=args.train_csv,
        tokenizer=tokenizer,
        group2id=group2id,
        max_length=args.max_length,
        text_template=args.text_template,
        encoding=args.csv_encoding,
    )
    eval_dataset = YelpAuthorStyleDataset(
        csv_path=args.eval_csv,
        tokenizer=tokenizer,
        group2id=group2id,
        max_length=args.max_length,
        text_template=args.text_template,
        encoding=args.csv_encoding,
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

    style_texts = args.group_prompts


    confounder_embeddings = embed_sentences(
        encoder=encoder,
        tokenizer=tokenizer,
        sentences=style_texts,
        device=device,
        max_length=args.max_length,
    )
    print(f"混杂变量嵌入向量形状: {confounder_embeddings.shape} (应该是 (2, hidden_size))")

    label_values = sorted(set(train_dataset.labels.tolist()) | set(eval_dataset.labels.tolist()))
    num_labels = len(label_values)
    if label_values != list(range(num_labels)):
        raise ValueError(
            "label 需从 0 开始连续编号，请先调整数据或在代码中添加映射。当前 label 集合为: "
            + str(label_values)
        )

    if len(args.label_prompts) != num_labels:
        raise ValueError(
            f"label_prompts 数量（{len(args.label_prompts)}）需要与 label 种类数（{num_labels}）一致。"
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
        num_style_classes = 2  # Hemingway 和 Shakespeare
        probe = load_probe(args.confounder_head_path, hidden_size, num_style_classes, device)

    style_embeddings = F.normalize(confounder_embeddings, p=2, dim=1)

    print("Estimating style-specific scales on training data...")
    scales = compute_scale(
        encoder=encoder,
        confounder_embeddings=confounder_embeddings,
        training_loader=train_loader,
        device=device,
        use_true_confounder=use_true_confounder,
        partial_a=args.partial_a,
        probe=probe,
    )

    classifier: Optional[LinearProbe] = None
    if train_linear_head_flag:
        print("Collecting debiased training features for linear head...")
        train_features, train_labels_tensor, train_styles_tensor = collect_debiased_embeddings(
            encoder=encoder,
            dataloader=train_loader,
            device=device,
            style_embeddings=style_embeddings,
            scales=scales,
            use_true_confounder=use_true_confounder,
            partial_a=args.partial_a,
            probe=probe,
            description="Collecting train features",
            debias_mode=args.debias_mode,           # [MOD]
            debias_strength=args.debias_strength,   # [MOD]
        )
        classifier = train_linear_head(
            features=train_features,
            labels=train_labels_tensor,
            styles=train_styles_tensor,
            device=device,
            num_labels=num_labels,
            epochs=args.head_epochs,
            lr=args.head_learning_rate,
            weight_decay=args.head_weight_decay,
            batch_size=args.head_batch_size,
            num_styles=style_embeddings.size(0),
            dro_eta=args.head_dro_eta,
            dro_gamma=args.head_dro_gamma,
            sampler_type=args.head_sampler,             # [MOD]
            dro_ema_alpha=args.head_dro_ema,            # [MOD]
            warmup_epochs=args.head_warmup_epochs,      # [MOD]
        )
        with torch.no_grad():
            train_logits = classifier(train_features.to(device))
            train_pred = train_logits.argmax(dim=1).cpu()
            train_acc = (train_pred == train_labels_tensor).float().mean().item()
        print(f"线性分类头训练集准确率: {train_acc:.4f}")
        del train_features, train_labels_tensor, train_styles_tensor, train_logits, train_pred

    try:
        id2group = {
            idx: (int(name) if isinstance(name, str) and name.isdigit() else name)
            for idx, name in id2group_raw.items()
        }
    except AttributeError:
        id2group = id2group_raw

    id2confounder_friendly = {
        idx: get_group_description(idx) for idx in range(len(group2id))
    }

    print("Starting evaluation...")
    print("=" * 80)
    print("数据集信息：")
    print(f"  - 标签：0=negative (负面), 1=positive (正面)")
    print(f"  - 混杂变量（写作风格）：Hemingway, Shakespeare")
    print(f"  - 四种情况：")
    print(f"    1. label=0, style=Hemingway (group_id=0)")
    print(f"    2. label=0, style=Shakespeare (group_id=1)")
    print(f"    3. label=1, style=Hemingway (group_id=2)")
    print(f"    4. label=1, style=Shakespeare (group_id=3)")
    print("=" * 80)
    run_epoch(
        encoder=encoder,
        dataloader=eval_loader,
        device=device,
        use_true_confounder=use_true_confounder,
        partial_a=args.partial_a,
        confounder_embeddings=confounder_embeddings,
        label_embeddings=label_embeddings,
        scales=scales,
        probe=probe,
        classifier=classifier,
        id2confounder=id2confounder_friendly,
        label_texts=args.label_prompts,
        debias_mode=args.debias_mode,
        debias_strength=args.debias_strength,
    )


if __name__ == "__main__":
    main()
