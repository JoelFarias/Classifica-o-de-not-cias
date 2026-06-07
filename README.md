# Classificação de Notícias por Tema

Projeto de NLP para classificação automática de notícias usando o dataset **AG News Classification Dataset**.

## Dataset

Dataset: [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

O dataset possui notícias classificadas em quatro categorias:

| Código | Classe |
| --- | --- |
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

Arquivos originais esperados:

- `data/raw/train.csv`
- `data/raw/test.csv`

Os arquivos CSV do dataset não são versionados no Git, pois são arquivos grandes. Eles devem ser baixados do Kaggle e colocados em `data/raw/`.

## Estrutura

```text
data/
  raw/
  processed/
models/
notebooks/
reports/
  figures/
src/
  data/
```

## Scripts da Pessoa 1

```bash
python src/data/load_ag_news.py
python src/data/inspect_ag_news.py
python src/data/explore_ag_news.py
python src/data/preprocess_ag_news.py
```

## Sprints Concluídas

| Sprint | Entrega |
| --- | --- |
| Sprint 1 | Estrutura do projeto e carregamento do dataset |
| Sprint 2 | Conferência de colunas, nulos, duplicados e classes |
| Sprint 3 | Análise exploratória, gráficos e exemplos |
| Sprint 4 | Pré-processamento e divisão treino/validação/teste |

## Dados Processados

O script de pré-processamento gera:

- `data/processed/train_processed.csv`
- `data/processed/validation_processed.csv`
- `data/processed/test_processed.csv`
- `data/processed/metadata.json`

Os CSVs processados também não são versionados, mas podem ser recriados com:

```bash
python src/data/preprocess_ag_news.py
```

## Próxima Etapa

A próxima parte do projeto é criar o baseline da Pessoa 2 usando TF-IDF com Logistic Regression, Naive Bayes ou Linear SVM.
