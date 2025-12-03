import argparse
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


torch.set_num_threads(5)
torch.set_num_interop_threads(5)


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


def encode_text_batch(encoder: AutoModel, batch: dict, device: torch.device) -> torch.Tensor:
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


class SemEvalSentimentDataset(Dataset):
    """针对 SemEval Laptop14 数据集的文本情感数据集封装。"""

    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer,
        aspect2id: Dict[str, int],
        max_length: int = 256,
        text_template: str = "Aspect: {aspect}. Sentence: {sentence}",
    ) -> None:
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} 不存在，请确认 SemEvalLaptop14 数据已放置在指定目录。")

        data = pd.read_csv(csv_path)
        required_columns = {"sentence", "aspect", "label_id"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"CSV 文件缺少必要列：{missing_columns}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_template = text_template
        self.aspect2id = aspect2id

        sentences = data["sentence"].astype(str).tolist()
        aspects = data["aspect"].astype(str).tolist()
        label_ids = data["label_id"].astype(int).to_numpy()

        try:
            confounder_ids = np.array([aspect2id[a] for a in aspects], dtype=int)
        except KeyError as exc:
            raise ValueError(f"发现未在映射中出现的 aspect：{exc.args[0]}") from exc

        formatted_texts = [text_template.format(sentence=s, aspect=a) for s, a in zip(sentences, aspects)]

        self.original_sentences = sentences
        self.aspects = aspects
        self.texts = formatted_texts
        self.labels = label_ids
        self.confounders = confounder_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "confounder": torch.tensor(self.confounders[idx], dtype=torch.long),
            "sentence": self.original_sentences[idx],
            "aspect": self.aspects[idx],
            "text": self.texts[idx],
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def inference_a(embeddings: torch.Tensor, confounder_embeddings: torch.Tensor) -> torch.Tensor:
    logits = torch.mm(embeddings, confounder_embeddings.t())
    probs = logits.softmax(dim=1)
    _, prediction = torch.max(probs, dim=1)
    return prediction


def supervised_inference_a(
    embeddings: torch.Tensor,
    probe: Optional[LinearProbe],
) -> torch.Tensor:
    if probe is None:
        raise ValueError("partial_a=True 时必须提供训练好的混杂变量分类器。")
    probe.eval()
    with torch.no_grad():
        logits = probe(embeddings)
        _, prediction = torch.max(logits, dim=1)
    return prediction


def build_aspect_mapping(train_csv: str, test_csv: str) -> Dict[str, int]:
    if not os.path.exists(train_csv):
        raise ValueError(f"未找到训练集文件：{train_csv}")
    if not os.path.exists(test_csv):
        raise ValueError(f"未找到测试集文件：{test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "aspect" not in train_df.columns or "aspect" not in test_df.columns:
        raise ValueError("train/test CSV 文件中缺少 aspect 列")

    aspects = pd.concat([train_df["aspect"], test_df["aspect"]], axis=0).astype(str)
    unique_aspects = sorted(aspects.unique())
    if not unique_aspects:
        raise ValueError("在数据集中没有发现任何 aspect")

    return {aspect: idx for idx, aspect in enumerate(unique_aspects)}


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
    num_groups = confounder_embeddings.size(0)
    scale_values: List[List[float]] = [[] for _ in range(num_groups)]

    for batch in tqdm(training_loader, desc="Computing Scale"):
        with torch.no_grad():
            embeddings = encode_text_batch(encoder, batch, device)

            if use_true_confounder:
                sensitive = batch["confounder"].to(device)
            else:
                if partial_a:
                    sensitive = supervised_inference_a(embeddings, probe)
                else:
                    sensitive = inference_a(embeddings, confounder_embeddings)

            for group_id in range(num_groups):
                mask = sensitive == group_id
                if mask.any():
                    group_embeddings = embeddings[mask]
                    inner = torch.mm(group_embeddings, confounder_embeddings[group_id].unsqueeze(1)).squeeze(1)
                    scale_values[group_id].extend(inner.detach().cpu().numpy().tolist())

    mean_values = [float(np.mean(values)) if values else 0.0 for values in scale_values]
    for idx, value in enumerate(mean_values):
        print(f"group{idx} scale: {value:.4f}")

    return torch.tensor(mean_values, device=device)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def test_epoch(
    encoder: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    use_true_confounder: bool,
    partial_a: bool,
    confounder_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    training_loader: DataLoader,
    probe: Optional[LinearProbe] = None,
    id2confounder: Optional[Dict[int, str]] = None,
    label_texts: Optional[Sequence[str]] = None,
) -> None:
    scales = compute_scale(
        encoder,
        confounder_embeddings,
        training_loader,
        device,
        use_true_confounder,
        partial_a,
        probe,
    )

    encoder.eval()

    predictions: List[int] = []
    labels_all: List[int] = []
    confounders_all: List[int] = []
    num_confounders = confounder_embeddings.size(0)
    num_labels = label_embeddings.size(0)

    correct_counts = np.zeros((num_labels, num_confounders), dtype=float)
    total_counts = np.zeros((num_labels, num_confounders), dtype=float)

    for batch in tqdm(dataloader, desc="Zero Shot Testing"):
        with torch.no_grad():
            labels = batch["label"].to(device)
            confounders = batch["confounder"].to(device)
            embeddings = encode_text_batch(encoder, batch, device)

            if use_true_confounder:
                sensitive = confounders
            else:
                if partial_a:
                    sensitive = supervised_inference_a(embeddings, probe)
                else:
                    sensitive = inference_a(embeddings, confounder_embeddings)

            for group_id in range(num_confounders):
                mask = sensitive == group_id
                if mask.any():
                    embeddings[mask] -= scales[group_id] * confounder_embeddings[group_id]

            logits = torch.mm(embeddings, label_embeddings.t())
            probs = logits.softmax(dim=1)
            _, pred = torch.max(probs, dim=1)

            predictions.extend(pred.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            confounders_all.extend(confounders.detach().cpu().numpy().tolist())

            labels_cpu = labels.detach().cpu()
            confounders_cpu = confounders.detach().cpu()
            pred_cpu = pred.detach().cpu()

            for label_idx in range(num_labels):
                for group_idx in range(num_confounders):
                    mask = ((labels_cpu == label_idx) & (confounders_cpu == group_idx))
                    if mask.any():
                        matches = (pred_cpu[mask] == label_idx).sum().item()
                        total = mask.sum().item()
                        correct_counts[label_idx, group_idx] += matches
                        total_counts[label_idx, group_idx] += total

    overall_acc = accuracy_score(labels_all, predictions)
    print(f"Overall accuracy: {overall_acc:.4f}")

    for label_idx in range(num_labels):
        label_accs = []
        for group_idx in range(num_confounders):
            acc = safe_div(correct_counts[label_idx, group_idx], total_counts[label_idx, group_idx])
            label_name = label_texts[label_idx] if label_texts else label_idx
            group_name = id2confounder[group_idx] if id2confounder else group_idx
            print(f"Accuracy for label={label_name}, confounder={group_name}: {acc:.4f}")
            if total_counts[label_idx, group_idx] > 0:
                label_accs.append(acc)
        if label_accs:
            print(f"Label {label_idx} accuracy gap (max-min): {max(label_accs) - min(label_accs):.4f}")

    group_metrics = {}
    for group_idx in range(num_confounders):
        indices = [i for i, g in enumerate(confounders_all) if g == group_idx]
        if not indices:
            continue
        preds_group = [predictions[i] for i in indices]
        labels_group = [labels_all[i] for i in indices]
        group_acc = accuracy_score(labels_group, preds_group)
        group_name = id2confounder[group_idx] if id2confounder else group_idx
        group_metrics[group_name] = group_acc
        print(f"Group {group_name} accuracy: {group_acc:.4f}")

    if group_metrics:
        values = list(group_metrics.values())
        print(f"Group accuracy gap (max-min): {max(values) - min(values):.4f}")

    positive_label_idx = num_labels - 1
    positive_rates = {}
    for group_idx in range(num_confounders):
        indices = [i for i, g in enumerate(confounders_all) if g == group_idx]
        if not indices:
            continue
        preds_group = [predictions[i] for i in indices]
        rate = preds_group.count(positive_label_idx) / len(preds_group)
        group_name = id2confounder[group_idx] if id2confounder else group_idx
        positive_rates[group_name] = rate
        print(f"Group {group_name} positive-rate: {rate:.4f}")

    if positive_rates:
        values = list(positive_rates.values())
        print(f"Demographic parity gap (positive label): {max(values) - min(values):.4f}")


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
    parser = argparse.ArgumentParser(description="TIE for 文本情感分析的实现")
    parser.add_argument("--train_csv", type=str, default="SemEvalLaptop14/train.csv", help="SemEval Laptop14 训练集 CSV 路径")
    parser.add_argument("--test_csv", type=str, default="SemEvalLaptop14/test.csv", help="SemEval Laptop14 测试集 CSV 路径")
    parser.add_argument(
        "--model_name",
        type=str,
        default=r"E:\paper_code_implements\MultiPerspectives_Sequentially_PredictionFeedback\src\pre-trained_model\bert_uncased_L-12_H-768_A-12",
        help="BERT模型路径（本地路径或Hugging Face模型名称）",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_true_confounder", action="store_true", help="若指定则直接使用 metadata 中的混杂标签")
    parser.add_argument("--partial_a", action="store_true", help="若指定则使用线性探针预测部分混杂变量")
    parser.add_argument("--confounder_head_path", type=str, default=None, help="混杂变量线性探针权重路径，仅在 partial_a=True 时使用")
    parser.add_argument(
        "--label_prompts",
        nargs="+",
        default=[
            "a laptop review expressing negative sentiment",
            "a laptop review expressing neutral sentiment",
            "a laptop review expressing positive sentiment",
        ],
        help="描述类别的提示文本，顺序需与 label_id 对应",
    )
    parser.add_argument(
        "--confounder_prompts",
        nargs="*",
        default=None,
        help="描述混杂变量（aspect）的提示文本，顺序需与 aspect 映射一致。不提供时将自动根据 aspect 生成。",
    )
    parser.add_argument(
        "--aspect_prompt_template",
        type=str,
        default="a laptop review focusing on the aspect: {aspect}",
        help="用于自动生成混杂变量提示的模板",
    )
    parser.add_argument(
        "--text_template",
        type=str,
        default="Aspect: {aspect}. Sentence: {sentence}",
        help="构造输入文本的模板",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 自动检测是本地路径还是Hugging Face模型名称
    model_path = args.model_name
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"从本地路径加载模型: {model_path}")
    else:
        print(f"从Hugging Face加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    encoder = AutoModel.from_pretrained(model_path)
    encoder = encoder.to(device)

    aspect2id = build_aspect_mapping(args.train_csv, args.test_csv)
    id2aspect = {idx: aspect for aspect, idx in aspect2id.items()}

    train_dataset = SemEvalSentimentDataset(
        args.train_csv,
        tokenizer,
        aspect2id,
        max_length=args.max_length,
        text_template=args.text_template,
    )
    test_dataset = SemEvalSentimentDataset(
        args.test_csv,
        tokenizer,
        aspect2id,
        max_length=args.max_length,
        text_template=args.text_template,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    confounder_texts: List[str]
    if args.confounder_prompts:
        if len(args.confounder_prompts) != len(aspect2id):
            raise ValueError(
                f"提供的 confounder_prompts 数量（{len(args.confounder_prompts)}）与 aspect 种类数（{len(aspect2id)}）不一致"
            )
        confounder_texts = list(args.confounder_prompts)
    else:
        confounder_texts = [
            args.aspect_prompt_template.format(aspect=id2aspect[idx]) for idx in range(len(aspect2id))
        ]

    confounder_embeddings = embed_sentences(
        encoder,
        tokenizer,
        confounder_texts,
        device,
        args.max_length,
    )

    label_ids = sorted(set(train_dataset.labels.tolist()) | set(test_dataset.labels.tolist()))
    num_labels = len(label_ids)
    if label_ids != list(range(num_labels)):
        raise ValueError(
            "label_id 需从 0 开始连续编号，请先调整数据或在代码中添加映射。当前 label_id 集合为: "
            + str(label_ids)
        )

    if len(args.label_prompts) != num_labels:
        raise ValueError(
            f"label_prompts 数量（{len(args.label_prompts)}）需要与 label_id 种类数（{num_labels}）一致"
        )

    label_embeddings = embed_sentences(
        encoder,
        tokenizer,
        args.label_prompts,
        device,
        args.max_length,
    )

    probe = None
    if args.partial_a:
        hidden_size = encoder.config.hidden_size
        probe = load_probe(args.confounder_head_path, hidden_size, confounder_embeddings.size(0), device)

    print("Starting evaluation...")
    test_epoch(
        encoder,
        test_loader,
        device,
        use_true_confounder=args.use_true_confounder,
        partial_a=args.partial_a,
        confounder_embeddings=confounder_embeddings,
        label_embeddings=label_embeddings,
        training_loader=train_loader,
        probe=probe,
        id2confounder=id2aspect,
        label_texts=args.label_prompts,
    )


if __name__ == "__main__":
    main()


