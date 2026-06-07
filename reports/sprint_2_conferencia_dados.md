# Sprint 2: Conferência e Entendimento dos Dados

## Objetivo

Entender a estrutura do dataset AG News e verificar se os dados estão corretos para seguir para a análise exploratória e modelagem.

## Script usado

A conferência foi feita com o script:

- `src/data/inspect_ag_news.py`

Esse script verifica:

- colunas dos arquivos;
- quantidade total de linhas;
- códigos das classes;
- valores vazios;
- linhas duplicadas;
- textos duplicados;
- distribuição das classes;
- repetição de textos entre treino e teste.

## Estrutura dos arquivos

Os arquivos `train.csv` e `test.csv` possuem as colunas principais:

| Coluna | Significado |
| --- | --- |
| `Class Index` | Código numérico da classe |
| `Title` | Título da notícia |
| `Description` | Descrição ou resumo da notícia |

Durante o carregamento dos dados, também foram criadas duas colunas auxiliares:

| Coluna | Significado |
| --- | --- |
| `class_name` | Nome da classe em texto |
| `text` | Junção de título e descrição |

## Mapeamento das classes

| Código | Classe |
| --- | --- |
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

## Resultado da conferência

| Verificação | Treino | Teste |
| --- | ---: | ---: |
| Total de linhas | 120.000 | 7.600 |
| Valores vazios | 0 | 0 |
| Linhas duplicadas completas | 0 | 0 |
| Textos duplicados | 0 | 0 |
| Códigos de classe inválidos | 0 | 0 |

Também foi verificado que não existem textos repetidos entre treino e teste.

## Distribuição das classes no treino

| Código | Classe | Quantidade |
| --- | --- | ---: |
| 1 | World | 30.000 |
| 2 | Sports | 30.000 |
| 3 | Business | 30.000 |
| 4 | Sci/Tech | 30.000 |

## Distribuição das classes no teste

| Código | Classe | Quantidade |
| --- | --- | ---: |
| 1 | World | 1.900 |
| 2 | Sports | 1.900 |
| 3 | Business | 1.900 |
| 4 | Sci/Tech | 1.900 |

## Conclusão da sprint

O dataset está correto para continuar o projeto. As colunas esperadas estão presentes, as classes estão balanceadas, não existem valores vazios, não existem duplicados e não há repetição de textos entre treino e teste.

A próxima etapa é a Sprint 3, que deve gerar gráficos, exemplos reais de cada categoria e análise do tamanho dos textos.
