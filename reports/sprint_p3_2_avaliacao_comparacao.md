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
| Épocas | 10 |
| Amostras de treino usadas | 40.000 |
| Tamanho máximo da sequência | 80 tokens |
| Tamanho do vocabulário | 30.000 |
| Embedding | 128 |
| Camada GRU | 96 unidades |
| Dispositivo | GPU CUDA |

## Resultado do GRU no teste

| Classe | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| World | 0,9070 | 0,8626 | 0,8843 |
| Sports | 0,9352 | 0,9421 | 0,9386 |
| Business | 0,8402 | 0,8358 | 0,8380 |
| Sci/Tech | 0,8235 | 0,8621 | 0,8424 |
| Média | 0,8765 | 0,8757 | 0,8758 |

**Acurácia do GRU no teste:** 87,57%.

## Baseline de referência

O baseline de referência é o LinearSVC da Pessoa 2.

| Modelo | Referência |
| --- | ---: |
| LinearSVC | F1 médio próximo de 0,91 a 0,92 |
| GRU | F1 médio de 0,8758 |

## Comparação

O GRU ficou abaixo do baseline LinearSVC, mas melhorou após o treinamento com GPU, 10 épocas e mais amostras. Isso mostra que o baseline tradicional ainda foi mais forte neste dataset, mas o modelo neural conseguiu um resultado competitivo.

Possíveis motivos:

- mesmo com 40.000 exemplos, o baseline TF-IDF com LinearSVC é muito forte para textos curtos;
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
| World | 1.639 | 73 | 96 | 92 |
| Sports | 45 | 1.790 | 24 | 41 |
| Business | 71 | 23 | 1.588 | 218 |
| Sci/Tech | 52 | 28 | 182 | 1.638 |

## Observação

O modelo GRU não superou o baseline TF-IDF nesta execução, mas melhorou bastante com GPU, mais épocas e mais dados. Ele cumpre o objetivo da Pessoa 3 por usar uma arquitetura de Deep Learning e permitir comparação direta.
