# Sprint 1: Estrutura do Projeto e Dataset

## Objetivo

Preparar a estrutura inicial do projeto e confirmar que os arquivos do dataset AG News estão disponíveis e podem ser carregados corretamente.

## Estrutura criada

- `data/raw/`: arquivos originais do dataset.
- `data/processed/`: arquivos tratados para modelagem.
- `notebooks/`: notebooks de exploração e experimentos.
- `src/`: scripts reutilizáveis do projeto.
- `src/data/`: scripts relacionados aos dados.
- `models/`: modelos treinados.
- `reports/`: relatórios e anotações do projeto.
- `reports/figures/`: gráficos gerados durante a análise.

## Dataset

Os arquivos do AG News foram colocados em `data/raw/`:

- `train.csv`
- `test.csv`

As colunas encontradas nos arquivos são:

- `Class Index`
- `Title`
- `Description`

## Validação feita

Foi criado o script `src/data/load_ag_news.py` para carregar e validar os arquivos.

Resultado da validação:

| Conjunto | Quantidade de linhas | Valores vazios |
| --- | ---: | ---: |
| Treino | 120.000 | 0 |
| Teste | 7.600 | 0 |

Distribuição das classes no treino:

| Classe | Quantidade |
| --- | ---: |
| Business | 30.000 |
| Sci/Tech | 30.000 |
| Sports | 30.000 |
| World | 30.000 |

Distribuição das classes no teste:

| Classe | Quantidade |
| --- | ---: |
| Business | 1.900 |
| Sci/Tech | 1.900 |
| Sports | 1.900 |
| World | 1.900 |

## Conclusão da sprint

A estrutura inicial do projeto foi criada e os arquivos do dataset foram carregados corretamente. O dataset está balanceado e não possui valores vazios nas colunas principais. A próxima etapa é a Sprint 2, com conferência mais detalhada dos dados, duplicados e estrutura das classes.
