# Sprint 1 (Pessoa 2): Recebimento e Verificação dos Dados

## Objetivo

Receber os dados preparados pela Pessoa 1 e confirmar que estão no formato correto para o treinamento do modelo baseline.

## Arquivos recebidos

| Arquivo | Uso |
| --- | --- |
| `data/processed/train_processed.csv` | Treino do baseline |
| `data/processed/validation_processed.csv` | Validação durante o desenvolvimento |
| `data/processed/test_processed.csv` | Avaliação final |
| `data/processed/metadata.json` | Resumo da divisão e das classes |

## Verificação feita

As colunas necessárias para o baseline estavam presentes em todos os arquivos:

| Coluna | Descrição |
| --- | --- |
| `clean_text` | Texto limpo com título + descrição |
| `label` | Classe ajustada de 0 a 3 |
| `class_name` | Nome legível da classe |

Conferência dos conjuntos:

| Conjunto | Linhas | Valores vazios em `clean_text` | Valores vazios em `label` |
| --- | ---: | ---: | ---: |
| Treino | 108.000 | 0 | 0 |
| Validação | 12.000 | 0 | 0 |
| Teste | 7.600 | 0 | 0 |

Distribuição das classes no treino:

| Classe | Quantidade |
| --- | ---: |
| World | 27.000 |
| Sports | 27.000 |
| Business | 27.000 |
| Sci/Tech | 27.000 |

## Mapeamento das classes

| Label | Nome |
| ---: | --- |
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## Conclusão da sprint

Os dados estão no formato esperado. A coluna `clean_text` será usada como entrada do TF-IDF e a coluna `label` como saída do modelo. A próxima etapa é criar e treinar o modelo baseline.
