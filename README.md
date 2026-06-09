# Classificação de Notícias por Tema

Projeto de NLP para classificação automática de notícias usando o dataset **AG News Classification Dataset**.

**Equipe:** Bruno Pereira, Joel Carolino e Gabriel César

## Responsáveis

| Pessoa | Integrante | Parte do projeto |
| --- | --- | --- |
| Pessoa 1 | Joel Carolino | Dados, análise exploratória e pré-processamento |
| Pessoa 2 | Gabriel César | Baseline, métricas e análise de erros |
| Pessoa 3 | Bruno Pereira | Deep Learning, comparação final e demonstração |

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
  models/
```

## Como rodar o projeto

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Baixar o dataset

Baixe os arquivos `train.csv` e `test.csv` do [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) e coloque em `data/raw/`.

### 3. Pré-processar os dados (Pessoa 1 - Joel)

```bash
python src/data/preprocess_ag_news.py
```

Isso gera os arquivos processados em `data/processed/`.

### 4. Treinar o baseline (Pessoa 2 - Gabriel)

```bash
python src/models/baseline_tfidf.py
```

Isso treina o TF-IDF + Logistic Regression e o TF-IDF + LinearSVC, avalia nos dados de teste e salva os modelos em `models/` e os resultados em `reports/baseline_results.csv`.

### 5. Treinar o modelo de Deep Learning (Pessoa 3 - Bruno)

```bash
python src/models/deep_learning_gru.py
```

Isso treina um modelo GRU em PyTorch, avalia nos dados de teste, salva métricas em `reports/` e salva o modelo em `models/deep_learning_gru.pt`.

### 6. Abrir a demonstração

```bash
streamlit run src/app_streamlit.py
```

### 7. Alternativa com DistilBERT

Também existe uma versão corrigida da proposta do Bruno usando DistilBERT:

```bash
python src/models/deep_learning_distilbert.py
streamlit run src/app_distilbert.py
```

Essa versão usa `data/processed/train_processed.csv`, `validation_processed.csv` e `test_processed.csv`.

## Scripts disponíveis

### Pessoa 1 — Joel — Dados e Pré-processamento

| Script | O que faz |
| --- | --- |
| `src/data/load_ag_news.py` | Carrega e valida os arquivos originais |
| `src/data/inspect_ag_news.py` | Inspeciona colunas, nulos e duplicados |
| `src/data/explore_ag_news.py` | Análise exploratória e geração de gráficos |
| `src/data/preprocess_ag_news.py` | Limpeza, divisão e salvamento dos dados processados |

### Pessoa 2 — Gabriel — Baseline

| Script | O que faz |
| --- | --- |
| `src/models/baseline_tfidf.py` | Treina TF-IDF + Logistic Regression e LinearSVC, avalia e salva resultados |

### Pessoa 3 — Bruno — Deep Learning e Demo

| Script | O que faz |
| --- | --- |
| `src/models/deep_learning_gru.py` | Treina e avalia o modelo GRU |
| `src/app_streamlit.py` | Abre a demo em Streamlit |
| `src/models/deep_learning_distilbert.py` | Alternativa com DistilBERT corrigida |
| `src/app_distilbert.py` | Demo opcional para o DistilBERT |

## Sprints Concluídas

### Pessoa 1 — Joel

| Sprint | Entrega | Report |
| --- | --- | --- |
| Sprint 1 | Estrutura do projeto e carregamento do dataset | `reports/sprint_1_estrutura_dataset.md` |
| Sprint 2 | Conferência de colunas, nulos, duplicados e classes | `reports/sprint_2_conferencia_dados.md` |
| Sprint 3 | Análise exploratória, gráficos e exemplos | `reports/sprint_3_analise_exploratoria.md` |
| Sprint 4 | Pré-processamento e divisão treino/validação/teste | `reports/sprint_4_preprocessamento_dados.md` |

### Pessoa 2 — Gabriel

| Sprint | Entrega | Report |
| --- | --- | --- |
| Sprint 1 | Recebimento e verificação dos dados processados | `reports/sprint_p2_1_recebimento_dados.md` |
| Sprint 2 | Criação e treinamento do baseline com TF-IDF | `reports/sprint_p2_2_treinamento_baseline.md` |
| Sprint 3 | Métricas, avaliação e matrizes de confusão | `reports/sprint_p2_3_metricas_avaliacao.md` |
| Sprint 4 | Análise de erros e resultados finais | `reports/sprint_p2_4_analise_erros.md` |

### Pessoa 3 — Bruno

| Sprint | Entrega | Report |
| --- | --- | --- |
| Sprint 1 | Modelo GRU de Deep Learning | `reports/sprint_p3_1_modelo_deep_learning.md` |
| Sprint 2 | Avaliação e comparação com baseline | `reports/sprint_p3_2_avaliacao_comparacao.md` |
| Sprint 3 | Demo em Streamlit | `reports/sprint_p3_3_demo_streamlit.md` |
| Sprint 4 | Resultado final da Pessoa 3 | `reports/sprint_p3_4_resultado_final.md` |

## Resultados do Baseline

O melhor modelo baseline foi o **LinearSVC** com acurácia de **91,4%** no conjunto de teste.

| Classe | F1-score |
| --- | ---: |
| World | 0,91 |
| Sports | 0,97 |
| Business | 0,88 |
| Sci/Tech | 0,89 |
| **Média** | **0,91** |

Os resultados completos estão em `reports/baseline_results.csv`.

## Resultado do Deep Learning

O modelo de Deep Learning da Pessoa 3 foi uma **GRU bidirecional** em PyTorch.

| Modelo | Acurácia | F1 médio |
| --- | ---: | ---: |
| LinearSVC baseline | 91,4% | 0,91 |
| GRU Deep Learning | 83,29% | 0,8330 |

O GRU não superou o baseline, mas completa a etapa de Deep Learning e permite comparar os dois tipos de abordagem.

## Dados Processados

Os CSVs processados não são versionados, mas podem ser recriados com:

```bash
python src/data/preprocess_ag_news.py
```

| Arquivo | Linhas |
| --- | ---: |
| `data/processed/train_processed.csv` | 108.000 |
| `data/processed/validation_processed.csv` | 12.000 |
| `data/processed/test_processed.csv` | 7.600 |

## Próxima Etapa

A próxima etapa é consolidar o relatório final e revisar a apresentação do grupo.
