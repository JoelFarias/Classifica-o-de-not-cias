import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_PATH = PROCESSED_DIR / "train_processed.csv"
VAL_PATH = PROCESSED_DIR / "validation_processed.csv"
TEST_PATH = PROCESSED_DIR / "test_processed.csv"
MODEL_PATH = MODELS_DIR / "deep_learning_gru.pt"
RESULTS_PATH = REPORTS_DIR / "deep_learning_results.csv"
CONFUSION_PATH = REPORTS_DIR / "deep_learning_confusion_matrix.csv"
SUMMARY_PATH = REPORTS_DIR / "deep_learning_run_summary.json"
CONFUSION_FIGURE_PATH = FIGURES_DIR / "deep_learning_confusion_matrix.png"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = pd.read_csv(path)
    required_columns = {"clean_text", "label", "class_name"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing)}")

    return df[["clean_text", "label", "class_name"]].copy()


def stratified_sample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df.reset_index(drop=True)

    samples_per_class = max_samples // df["label"].nunique()
    sampled_groups = []
    for _, group in df.groupby("label"):
        sampled_groups.append(group.sample(min(len(group), samples_per_class), random_state=seed))

    sampled = pd.concat(sampled_groups).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def build_vocab(texts: pd.Series, max_vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)[:max_length]]
    if len(ids) < max_length:
        ids.extend([vocab[PAD_TOKEN]] * (max_length - len(ids)))
    return ids


class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict[str, int], max_length: int):
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
        encoded = [encode_text(text, vocab, max_length) for text in df["clean_text"]]
        self.inputs = torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, split_name: str) -> dict:
    y_true, y_pred = predict(model, dataloader, device)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n=== {split_name} ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(f"Acuracia: {accuracy:.4f}")
    return {"report": report, "accuracy": accuracy, "y_true": y_true, "y_pred": y_pred}


def save_results(results: dict[str, dict]) -> None:
    rows = []
    for model_name, result in results.items():
        for class_or_avg, metrics in result["report"].items():
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "modelo": model_name,
                        "classe": class_or_avg,
                        "precision": round(metrics.get("precision", 0), 4),
                        "recall": round(metrics.get("recall", 0), 4),
                        "f1-score": round(metrics.get("f1-score", 0), 4),
                        "support": metrics.get("support", ""),
                    }
                )
        rows.append(
            {
                "modelo": model_name,
                "classe": "accuracy",
                "precision": "",
                "recall": "",
                "f1-score": round(result["accuracy"], 4),
                "support": "",
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(CONFUSION_PATH)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusao - GRU")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(CONFUSION_FIGURE_PATH, dpi=160)
    plt.close()


def save_model(model: nn.Module, vocab: dict[str, int], config: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": config,
            "class_names": CLASS_NAMES,
        },
        MODEL_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina um modelo GRU para classificar noticias AG News.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-train-samples", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_df = stratified_sample(load_split(TRAIN_PATH), args.max_train_samples, args.seed)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    print(f"Treino usado: {len(train_df)} linhas")
    print(f"Validacao: {len(val_df)} linhas")
    print(f"Teste: {len(test_df)} linhas")

    vocab = build_vocab(train_df["clean_text"], args.max_vocab_size)
    print(f"Vocabulario: {len(vocab)} tokens")

    train_dataset = NewsDataset(train_df, vocab, args.max_length)
    val_dataset = NewsDataset(val_df, vocab, args.max_length)
    test_dataset = NewsDataset(test_df, vocab, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    config = {
        "vocab_size": len(vocab),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": len(CLASS_NAMES),
        "dropout": args.dropout,
        "max_length": args.max_length,
        "max_train_samples": len(train_df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }

    model = GRUClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, device, f"Validacao - epoca {epoch}")
        history.append({"epoch": epoch, "train_loss": loss, "val_accuracy": val_result["accuracy"]})
        print(f"Epoca {epoch}: loss={loss:.4f}, val_accuracy={val_result['accuracy']:.4f}")

    test_result = evaluate(model, test_loader, device, "Teste final")
    save_results({"gru_val": val_result, "gru_test": test_result})
    save_confusion_matrix(test_result["y_true"], test_result["y_pred"])
    save_model(model, vocab, config)

    summary = {
        "model": "Bidirectional GRU",
        "device": str(device),
        "history": history,
        "test_accuracy": round(test_result["accuracy"], 4),
        "results_file": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "confusion_matrix_file": str(CONFUSION_PATH.relative_to(PROJECT_ROOT)),
        "confusion_matrix_figure": str(CONFUSION_FIGURE_PATH.relative_to(PROJECT_ROOT)),
        "model_file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "config": config,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResultados salvos em: {RESULTS_PATH}")
    print(f"Matriz de confusao salva em: {CONFUSION_PATH}")
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
