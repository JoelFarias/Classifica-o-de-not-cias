from load_ag_news import CLASS_NAMES, EXPECTED_COLUMNS, TEST_PATH, TRAIN_PATH, load_dataset


def class_distribution(df):
    distribution = (
        df.groupby(["Class Index", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values("Class Index")
    )
    return distribution


def inspect_split(name, df):
    print(f"\n{name}")
    print("=" * len(name))
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {list(df.columns)}")
    print(f"Colunas esperadas presentes: {all(col in df.columns for col in EXPECTED_COLUMNS)}")
    print(f"Codigos de classe encontrados: {sorted(df['Class Index'].unique().tolist())}")
    print(f"Codigos de classe invalidos: {df['class_name'].isna().sum()}")
    print(f"Valores vazios por coluna:\n{df.isna().sum()}")
    print(f"Linhas duplicadas completas: {df.duplicated().sum()}")
    print(f"Textos duplicados: {df['text'].duplicated().sum()}")
    print("\nDistribuicao por classe:")
    print(class_distribution(df).to_string(index=False))


def main():
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    inspect_split("Treino", train_df)
    inspect_split("Teste", test_df)

    train_texts = set(train_df["text"])
    test_texts = set(test_df["text"])
    overlap = train_texts.intersection(test_texts)

    print("\nComparacao treino vs teste")
    print("========================")
    print(f"Textos repetidos entre treino e teste: {len(overlap)}")
    print("\nMapeamento das classes:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_id} = {class_name}")


if __name__ == "__main__":
    main()
