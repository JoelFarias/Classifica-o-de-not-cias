import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_PATH = PROCESSED_DIR / "train_processed.csv"
VAL_PATH = PROCESSED_DIR / "validation_processed.csv"
TEST_PATH = PROCESSED_DIR / "test_processed.csv"

MODEL_DIR = MODELS_DIR / "distilbert_news_model"
RESULTS_PATH = REPORTS_DIR / "distilbert_results.csv"
CONFUSION_PATH = REPORTS_DIR / "distilbert_confusion_matrix.csv"
SUMMARY_PATH = REPORTS_DIR / "distilbert_run_summary.json"
CONFUSION_FIGURE_PATH = FIGURES_DIR / "distilbert_confusion_matrix.png"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = pd.read_csv(path)
    required_columns = {"clean_text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing)}")

    return df[["clean_text", "label"]].dropna().copy()


def stratified_sample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df.reset_index(drop=True)

    samples_per_class = max_samples // df["label"].nunique()
    sampled_groups = []
    for _, group in df.groupby("label"):
        sampled_groups.append(group.sample(min(len(group), samples_per_class), random_state=seed))

    return pd.concat(sampled_groups).sample(frac=1, random_state=seed).reset_index(drop=True)


class NewsTransformerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int):
        self.texts = df["clean_text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def freeze_base_model(model: AutoModelForSequenceClassification) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("classifier") and not name.startswith("pre_classifier"):
            parameter.requires_grad = False


def train_one_epoch(model, dataloader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


def predict(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def evaluate(model, dataloader, device: torch.device, split_name: str) -> dict:
    y_true, y_pred = predict(model, dataloader, device)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n=== {split_name} ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
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
    ax.set_title("Matriz de confusao - DistilBERT")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina DistilBERT para classificar noticias AG News.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine-tune-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_df = stratified_sample(load_split(TRAIN_PATH), args.max_train_samples, args.seed)
    val_df = stratified_sample(load_split(VAL_PATH), args.max_val_samples, args.seed)
    test_df = stratified_sample(load_split(TEST_PATH), args.max_test_samples, args.seed)

    print(f"Treino usado: {len(train_df)} linhas")
    print(f"Validacao: {len(val_df)} linhas")
    print(f"Teste usado: {len(test_df)} linhas")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(CLASS_NAMES))

    if not args.fine_tune_all:
        freeze_base_model(model)

    model.to(device)

    train_dataset = NewsTransformerDataset(train_df, tokenizer, args.max_length)
    val_dataset = NewsTransformerDataset(val_df, tokenizer, args.max_length)
    test_dataset = NewsTransformerDataset(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    history = []
    val_result = None
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_result = evaluate(model, val_loader, device, f"Validacao - epoca {epoch}")
        history.append({"epoch": epoch, "train_loss": loss, "val_accuracy": val_result["accuracy"]})
        print(f"Epoca {epoch}: loss={loss:.4f}, val_accuracy={val_result['accuracy']:.4f}")

    test_result = evaluate(model, test_loader, device, "Teste final")
    save_results({"distilbert_val": val_result, "distilbert_test": test_result})
    save_confusion_matrix(test_result["y_true"], test_result["y_pred"])

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    summary = {
        "model": "DistilBERT",
        "base_model": args.model_name,
        "device": str(device),
        "history": history,
        "test_accuracy": round(test_result["accuracy"], 4),
        "frozen_base_model": not args.fine_tune_all,
        "results_file": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "confusion_matrix_file": str(CONFUSION_PATH.relative_to(PROJECT_ROOT)),
        "confusion_matrix_figure": str(CONFUSION_FIGURE_PATH.relative_to(PROJECT_ROOT)),
        "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "config": vars(args),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nModelo salvo em: {MODEL_DIR}")
    print(f"Resultados salvos em: {RESULTS_PATH}")
    print(f"Matriz salva em: {CONFUSION_PATH}")


if __name__ == "__main__":
    main()
