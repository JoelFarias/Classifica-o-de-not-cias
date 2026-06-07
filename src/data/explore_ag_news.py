from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from load_ag_news import TEST_PATH, TRAIN_PATH, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["title_word_count"] = df["Title"].str.split().str.len()
    df["description_word_count"] = df["Description"].str.split().str.len()
    return df


def class_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_counts = train_df["class_name"].value_counts().rename("train")
    test_counts = test_df["class_name"].value_counts().rename("test")

    distribution = (
        pd.concat([train_counts, test_counts], axis=1)
        .fillna(0)
        .astype(int)
        .loc[["World", "Sports", "Business", "Sci/Tech"]]
    )
    return distribution


def text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("class_name")["word_count"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .loc[["World", "Sports", "Business", "Sci/Tech"]]
    )
    return stats


def save_class_distribution_plot(distribution: pd.DataFrame) -> None:
    ax = distribution.plot(kind="bar", figsize=(9, 5), color=["#2563eb", "#f97316"])
    ax.set_title("Distribuicao das classes no AG News")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de noticias")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Treino", "Teste"])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_train_test.png", dpi=160)
    plt.close()


def save_word_count_boxplot(train_df: pd.DataFrame) -> None:
    ordered = ["World", "Sports", "Business", "Sci/Tech"]
    data = [train_df.loc[train_df["class_name"] == class_name, "word_count"] for class_name in ordered]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(ordered)
    ax.set_title("Tamanho dos textos por classe no treino")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de palavras")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "word_count_by_class_boxplot.png", dpi=160)
    plt.close()


def save_word_count_histogram(train_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_df["word_count"], bins=40, color="#16a34a", edgecolor="white")
    ax.set_title("Distribuicao do tamanho dos textos no treino")
    ax.set_xlabel("Quantidade de palavras")
    ax.set_ylabel("Quantidade de noticias")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "word_count_histogram_train.png", dpi=160)
    plt.close()


def examples_by_class(train_df: pd.DataFrame) -> pd.DataFrame:
    ordered = ["World", "Sports", "Business", "Sci/Tech"]
    examples = []

    for class_name in ordered:
        sample = train_df.loc[train_df["class_name"] == class_name].head(3)
        for _, row in sample.iterrows():
            examples.append(
                {
                    "class_name": class_name,
                    "title": row["Title"],
                    "word_count": int(row["word_count"]),
                }
            )

    return pd.DataFrame(examples)


def find_possible_ambiguous_examples(train_df: pd.DataFrame) -> pd.DataFrame:
    patterns = [
        ("Business com tecnologia", "Business", ["software", "internet", "technology", "computer"]),
        ("Sci/Tech com empresas", "Sci/Tech", ["company", "market", "business", "microsoft"]),
        ("Sports com contexto internacional", "Sports", ["world", "olympic", "country", "international"]),
    ]

    examples = []
    lower_text = train_df["text"].str.lower()

    for note, class_name, keywords in patterns:
        class_mask = train_df["class_name"] == class_name
        keyword_mask = lower_text.apply(lambda text: any(keyword in text for keyword in keywords))
        sample = train_df.loc[class_mask & keyword_mask].head(2)

        for _, row in sample.iterrows():
            examples.append(
                {
                    "observacao": note,
                    "class_name": class_name,
                    "title": row["Title"],
                }
            )

    return pd.DataFrame(examples)


def write_examples_report(examples: pd.DataFrame, ambiguous: pd.DataFrame) -> None:
    lines = ["# Exemplos por Classe", ""]

    for class_name, group in examples.groupby("class_name", sort=False):
        lines.append(f"## {class_name}")
        lines.append("")
        for _, row in group.iterrows():
            lines.append(f"- {row['title']} ({row['word_count']} palavras)")
        lines.append("")

    lines.append("## Possiveis exemplos ambiguos")
    lines.append("")
    for _, row in ambiguous.iterrows():
        lines.append(f"- {row['observacao']}: {row['title']} [{row['class_name']}]")
    lines.append("")

    (REPORTS_DIR / "sprint_3_exemplos_por_classe.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_sprint_report(
    distribution: pd.DataFrame,
    stats: pd.DataFrame,
    shortest: pd.DataFrame,
    longest: pd.DataFrame,
    ambiguous: pd.DataFrame,
) -> None:
    report = f"""# Sprint 3: Analise Exploratoria

## Objetivo

Gerar informacoes visuais e exemplos para entender melhor o dataset AG News antes da modelagem.

## Arquivos gerados

- `reports/figures/class_distribution_train_test.png`
- `reports/figures/word_count_by_class_boxplot.png`
- `reports/figures/word_count_histogram_train.png`
- `reports/sprint_3_distribuicao_classes.csv`
- `reports/sprint_3_tamanho_textos.csv`
- `reports/sprint_3_exemplos_por_classe.md`

## Distribuicao das classes

{distribution.to_markdown()}

O dataset esta balanceado: cada classe possui a mesma quantidade de exemplos no treino e no teste.

## Tamanho dos textos no treino

Os textos foram medidos usando a coluna `text`, formada pela juncao de `Title` e `Description`.

{stats.to_markdown()}

## Textos mais curtos no treino

{shortest[["class_name", "word_count", "Title"]].to_markdown(index=False)}

## Textos mais longos no treino

{longest[["class_name", "word_count", "Title"]].to_markdown(index=False)}

## Possiveis exemplos ambiguos

{ambiguous.to_markdown(index=False)}

## Observacoes

- As classes estao perfeitamente balanceadas, o que facilita a avaliacao dos modelos.
- A maioria dos textos tem tamanho moderado, adequado para modelos tradicionais e modelos de Deep Learning.
- Existem noticias que podem confundir o modelo, principalmente quando misturam termos de negocios, tecnologia, esportes e contexto internacional.
- A coluna `text` ja esta pronta como entrada principal para os proximos passos, pois combina titulo e descricao.

## Conclusao da sprint

A analise exploratoria confirmou que o dataset e bem distribuido e possui textos de tamanho adequado para classificacao. A proxima etapa e a Sprint 4, com limpeza e preparacao dos dados para modelagem.
"""
    (REPORTS_DIR / "sprint_3_analise_exploratoria.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    train_df = add_text_features(load_dataset(TRAIN_PATH))
    test_df = add_text_features(load_dataset(TEST_PATH))

    distribution = class_distribution(train_df, test_df)
    stats = text_length_stats(train_df)
    examples = examples_by_class(train_df)
    ambiguous = find_possible_ambiguous_examples(train_df)

    shortest = train_df.nsmallest(5, "word_count")
    longest = train_df.nlargest(5, "word_count")

    distribution.to_csv(REPORTS_DIR / "sprint_3_distribuicao_classes.csv")
    stats.to_csv(REPORTS_DIR / "sprint_3_tamanho_textos.csv")

    save_class_distribution_plot(distribution)
    save_word_count_boxplot(train_df)
    save_word_count_histogram(train_df)
    write_examples_report(examples, ambiguous)
    write_sprint_report(distribution, stats, shortest, longest, ambiguous)

    print("Sprint 3 concluida.")
    print(f"Relatorio: {REPORTS_DIR / 'sprint_3_analise_exploratoria.md'}")
    print(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
