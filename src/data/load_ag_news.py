from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

CLASS_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

EXPECTED_COLUMNS = ["Class Index", "Title", "Description"]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = pd.read_csv(path)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing_columns}")

    df = df[EXPECTED_COLUMNS].copy()
    df["class_name"] = df["Class Index"].map(CLASS_NAMES)
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")

    return df


def summarize_dataset(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Linhas: {len(df)}")
    print(f"Valores vazios:\n{df.isna().sum()}")
    print("\nDistribuicao por classe:")
    print(df["class_name"].value_counts().sort_index())
    print("\nPrimeiras linhas:")
    print(df[["Class Index", "class_name", "Title"]].head())


def main() -> None:
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    summarize_dataset("Treino", train_df)
    summarize_dataset("Teste", test_df)


if __name__ == "__main__":
    main()
