# Dados Processados

Esta pasta guarda os arquivos tratados para modelagem.

Os arquivos são gerados pelo script:

- `src/data/preprocess_ag_news.py`

## Arquivos gerados

- `train_processed.csv`: parte de treino usada pelos modelos.
- `validation_processed.csv`: parte de validação separada do treino original.
- `test_processed.csv`: teste final original do dataset.
- `metadata.json`: resumo da divisão, classes e colunas.

## Colunas principais

| Coluna | Uso |
| --- | --- |
| `Class Index` | Classe original do dataset, de 1 a 4 |
| `label` | Classe ajustada para modelos, de 0 a 3 |
| `class_name` | Nome da classe |
| `text` | `Title + Description` sem limpeza adicional |
| `clean_text` | Texto limpo para modelagem |

## Divisão dos dados

O treino original é separado em treino e validação com divisão estratificada, mantendo a proporção das classes.
