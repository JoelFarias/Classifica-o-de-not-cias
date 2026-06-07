import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from load_ag_news import CLASS_NAMES, TEST_PATH, TRAIN_PATH, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
VALIDATION_PROCESSED_PATH = PROCESSED_DATA_DIR / "validation_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.json"

VALIDATION_SIZE = 0.10
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\\+", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"()/%$-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["label"] = prepared["Class Index"] - 1
    prepared["clean_text"] = prepared["text"].apply(clean_text)
    prepared["text_word_count"] = prepared["text"].str.split().str.len()
    prepared["clean_text_word_count"] = prepared["clean_text"].str.split().str.len()

    return prepared[
        [
            "Class Index",
            "label",
            "class_name",
            "Title",
            "Description",
            "text",
            "clean_text",
            "text_word_count",
            "clean_text_word_count",
        ]
    ]


def split_train_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split, validation_split = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["Class Index"],
    )

    return train_split.reset_index(drop=True), validation_split.reset_index(drop=True)


def distribution_table(df: pd.DataFrame) -> dict[str, int]:
    counts = df["class_name"].value_counts().sort_index()
    return {class_name: int(count) for class_name, count in counts.items()}


def write_metadata(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    metadata = {
        "dataset": "AG News Classification Dataset",
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
        "label_mapping": {
            "0": "World",
            "1": "Sports",
            "2": "Business",
            "3": "Sci/Tech",
        },
        "files": {
            "train": str(TRAIN_PROCESSED_PATH.relative_to(PROJECT_ROOT)),
            "validation": str(VALIDATION_PROCESSED_PATH.relative_to(PROJECT_ROOT)),
            "test": str(TEST_PROCESSED_PATH.relative_to(PROJECT_ROOT)),
        },
        "rows": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "class_distribution": {
            "train": distribution_table(train_df),
            "validation": distribution_table(validation_df),
            "test": distribution_table(test_df),
        },
        "text_columns": {
            "text": "Title + Description sem limpeza adicional",
            "clean_text": "Texto em minusculas, com barras invertidas removidas, espacos normalizados e caracteres estranhos tratados",
        },
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    raw_train_df = prepare_columns(load_dataset(TRAIN_PATH))
    raw_test_df = prepare_columns(load_dataset(TEST_PATH))
    train_df, validation_df = split_train_validation(raw_train_df)

    train_df.to_csv(TRAIN_PROCESSED_PATH, index=False)
    validation_df.to_csv(VALIDATION_PROCESSED_PATH, index=False)
    raw_test_df.to_csv(TEST_PROCESSED_PATH, index=False)
    write_metadata(train_df, validation_df, raw_test_df)

    print("Pre-processamento concluido.")
    print(f"Treino: {len(train_df)} linhas -> {TRAIN_PROCESSED_PATH}")
    print(f"Validacao: {len(validation_df)} linhas -> {VALIDATION_PROCESSED_PATH}")
    print(f"Teste: {len(raw_test_df)} linhas -> {TEST_PROCESSED_PATH}")
    print(f"Metadados: {METADATA_PATH}")


if __name__ == "__main__":
    main()
