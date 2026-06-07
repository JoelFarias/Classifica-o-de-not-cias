# Sprint 4: Pré-processamento e Dados Prontos

## Objetivo

Preparar os textos do dataset AG News para serem usados pelos modelos da Pessoa 2 e da Pessoa 3.

## Script usado

O pré-processamento foi feito com o script:

- `src/data/preprocess_ag_news.py`

Esse script:

- carrega `train.csv` e `test.csv`;
- cria a coluna `text` com `Title + Description`;
- cria a coluna `clean_text` com o texto limpo;
- cria a coluna `label` com classes de `0` a `3`;
- separa parte do treino original para validação;
- salva os arquivos processados em `data/processed/`;
- salva metadados em `data/processed/metadata.json`.

## Transformações feitas

Na coluna `clean_text`, foram aplicadas as seguintes transformações:

- conversão do texto para minúsculas;
- remoção de barras invertidas;
- remoção de entidades simples de HTML, como `&nbsp;`;
- remoção de caracteres estranhos;
- normalização de espaços extras;
- manutenção de pontuações úteis para o texto.

## Arquivos gerados

| Arquivo | Uso |
| --- | --- |
| `data/processed/train_processed.csv` | Treino dos modelos |
| `data/processed/validation_processed.csv` | Validação durante os testes |
| `data/processed/test_processed.csv` | Teste final |
| `data/processed/metadata.json` | Resumo da divisão e das classes |

## Colunas finais

| Coluna | Significado |
| --- | --- |
| `Class Index` | Classe original do dataset, de 1 a 4 |
| `label` | Classe ajustada para modelos, de 0 a 3 |
| `class_name` | Nome da classe |
| `Title` | Título original da notícia |
| `Description` | Descrição original da notícia |
| `text` | Junção de título e descrição |
| `clean_text` | Texto limpo para modelagem |
| `text_word_count` | Quantidade de palavras no texto original combinado |
| `clean_text_word_count` | Quantidade de palavras no texto limpo |

## Mapeamento das classes

| Classe original | Label para modelo | Nome |
| ---: | ---: | --- |
| 1 | 0 | World |
| 2 | 1 | Sports |
| 3 | 2 | Business |
| 4 | 3 | Sci/Tech |

## Divisão dos dados

O treino original foi separado em:

- 90% para treino;
- 10% para validação.

A separação foi estratificada, ou seja, manteve a mesma proporção das classes.

| Conjunto | Total de linhas |
| --- | ---: |
| Treino | 108.000 |
| Validação | 12.000 |
| Teste | 7.600 |

## Distribuição das classes

| Classe | Treino | Validação | Teste |
| --- | ---: | ---: | ---: |
| Business | 27.000 | 3.000 | 1.900 |
| Sci/Tech | 27.000 | 3.000 | 1.900 |
| Sports | 27.000 | 3.000 | 1.900 |
| World | 27.000 | 3.000 | 1.900 |

## Como usar os dados nas próximas partes

Para a Pessoa 2, responsável pelo baseline:

- usar `clean_text` como entrada do TF-IDF;
- usar `label` como saída do modelo;
- treinar com `train_processed.csv`;
- validar com `validation_processed.csv`;
- testar no final com `test_processed.csv`.

Para a Pessoa 3, responsável pelo Deep Learning:

- usar `clean_text` como entrada principal;
- usar `label` como classe;
- aplicar tokenização específica do modelo escolhido;
- manter `test_processed.csv` apenas para avaliação final.

## Conclusão da sprint

Os dados estão prontos para modelagem. A Pessoa 2 já pode criar o baseline e a Pessoa 3 já pode usar o mesmo formato para treinar o modelo de Deep Learning.
