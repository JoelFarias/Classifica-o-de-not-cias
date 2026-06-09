# Sprint 2 (Pessoa 3 - Bruno): Avaliação e Comparação

## Objetivo

Avaliar o modelo GRU e comparar os resultados com o melhor baseline da Pessoa 2.

## Arquivos gerados pelo script

Após executar:

```bash
python src/models/deep_learning_gru.py
```

São gerados:

| Arquivo | Conteúdo |
| --- | --- |
| `reports/deep_learning_results.csv` | Métricas do modelo GRU |
| `reports/deep_learning_confusion_matrix.csv` | Matriz de confusão |
| `reports/figures/deep_learning_confusion_matrix.png` | Figura da matriz de confusão |
| `reports/deep_learning_run_summary.json` | Resumo da execução |
| `models/deep_learning_gru.pt` | Modelo treinado |

## Configuração usada

| Parâmetro | Valor |
| --- | ---: |
| Épocas | 3 |
| Amostras de treino usadas | 20.000 |
| Tamanho máximo da sequência | 80 tokens |
| Tamanho do vocabulário | 20.000 |
| Embedding | 128 |
| Camada GRU | 96 unidades |

## Resultado do GRU no teste

| Classe | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| World | 0,8251 | 0,8589 | 0,8417 |
| Sports | 0,9091 | 0,8842 | 0,8965 |
| Business | 0,8057 | 0,7768 | 0,7910 |
| Sci/Tech | 0,7940 | 0,8116 | 0,8027 |
| Média | 0,8335 | 0,8329 | 0,8330 |

**Acurácia do GRU no teste:** 83,29%.

## Baseline de referência

O baseline de referência é o LinearSVC da Pessoa 2.

| Modelo | Referência |
| --- | ---: |
| LinearSVC | F1 médio próximo de 0,91 a 0,92 |
| GRU | F1 médio de 0,8330 |

## Comparação

O GRU ficou abaixo do baseline LinearSVC. Isso não invalida a entrega, porque o objetivo da Pessoa 3 era criar e avaliar um modelo de Deep Learning, mas mostra que o baseline tradicional ainda foi mais forte neste dataset.

Possíveis motivos:

- o GRU foi treinado com 20.000 exemplos para manter a execução leve;
- o TF-IDF com LinearSVC é um baseline muito forte para classificação de textos curtos;
- mais épocas, mais dados ou um modelo Transformer poderiam melhorar o resultado;
- Business e Sci/Tech continuam sendo as classes mais difíceis de separar.

## Como comparar

Comparar principalmente:

- acurácia;
- F1 médio;
- F1 por classe;
- erros entre Business e Sci/Tech;
- erros entre World e Business.

## Matriz de confusão do GRU

| Classe real | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| World | 1.632 | 90 | 109 | 69 |
| Sports | 116 | 1.680 | 34 | 70 |
| Business | 139 | 24 | 1.476 | 261 |
| Sci/Tech | 91 | 54 | 213 | 1.542 |

## Observação

O modelo GRU não superou o baseline TF-IDF nesta execução, principalmente por ter sido treinado com uma amostra menor para economizar tempo. Mesmo assim, ele cumpre o objetivo da Pessoa 3 por usar uma arquitetura de Deep Learning e permitir comparação direta.
