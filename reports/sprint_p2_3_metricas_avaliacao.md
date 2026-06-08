# Sprint 3 (Pessoa 2 - Gabriel): Métricas e Avaliação do Baseline

## Objetivo

Avaliar o desempenho dos modelos baseline nos conjuntos de validação e teste, calcular as métricas principais e gerar a matriz de confusão.

## Conjunto de avaliação

A avaliação final foi feita no conjunto de teste (`test_processed.csv`) com 7.600 exemplos, 1.900 por classe.

## Resultados no conjunto de teste

### Logistic Regression

**Acurácia geral: 90,7%**

| Classe | Precision | Recall | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| World | 0,89 | 0,92 | 0,91 | 1.900 |
| Sports | 0,97 | 0,97 | 0,97 | 1.900 |
| Business | 0,88 | 0,86 | 0,87 | 1.900 |
| Sci/Tech | 0,88 | 0,87 | 0,88 | 1.900 |
| **Média** | **0,91** | **0,91** | **0,91** | **7.600** |

### LinearSVC

**Acurácia geral: 91,4%**

| Classe | Precision | Recall | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| World | 0,90 | 0,93 | 0,91 | 1.900 |
| Sports | 0,97 | 0,97 | 0,97 | 1.900 |
| Business | 0,89 | 0,87 | 0,88 | 1.900 |
| Sci/Tech | 0,89 | 0,88 | 0,89 | 1.900 |
| **Média** | **0,91** | **0,91** | **0,91** | **7.600** |

## Resumo comparativo

| Métrica | Logistic Regression | LinearSVC |
| --- | ---: | ---: |
| Acurácia | 90,7% | 91,4% |
| F1 médio | 0,91 | 0,91 |
| Melhor classe | Sports (0,97) | Sports (0,97) |
| Pior classe | Business (0,87) | Business (0,88) |

## Matriz de confusão

### Logistic Regression

|  | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| **World** | 1.749 | 14 | 77 | 60 |
| **Sports** | 11 | 1.843 | 20 | 26 |
| **Business** | 68 | 7 | 1.637 | 188 |
| **Sci/Tech** | 54 | 12 | 181 | 1.653 |

### LinearSVC

|  | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| **World** | 1.759 | 12 | 73 | 56 |
| **Sports** | 10 | 1.848 | 18 | 24 |
| **Business** | 62 | 6 | 1.650 | 182 |
| **Sci/Tech** | 48 | 10 | 175 | 1.667 |

## Observações sobre as matrizes

- **Sports** é a classe mais fácil de classificar, com poucos erros em ambos os modelos.
- **Business e Sci/Tech** se confundem com frequência entre si. Isso é esperado porque notícias de tecnologia muitas vezes envolvem empresas e negócios.
- **World e Business** também apresentam alguma confusão, possivelmente por notícias sobre economia global ou crises internacionais.

## Arquivo de resultados

Os resultados foram salvos em:

| Arquivo | Conteúdo |
| --- | --- |
| `reports/baseline_results.csv` | Métricas de precisão, recall e F1 por classe |

## Conclusão da sprint

O LinearSVC foi ligeiramente superior ao Logistic Regression. Ambos os modelos baseline alcançaram acurácia acima de 90%, o que é uma referência sólida para comparação com o modelo de Deep Learning do Bruno Pereira (Pessoa 3). A maior dificuldade observada foi a separação entre as classes Business e Sci/Tech.
