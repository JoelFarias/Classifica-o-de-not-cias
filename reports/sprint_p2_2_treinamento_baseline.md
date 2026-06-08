# Sprint 2 (Pessoa 2): Criação e Treinamento do Baseline

## Objetivo

Transformar os textos em representação numérica com TF-IDF e treinar um modelo baseline simples de classificação.

## Script usado

O treinamento do baseline foi feito com o script:

- `src/models/baseline_tfidf.py`

## Representação dos textos com TF-IDF

O TF-IDF (Term Frequency–Inverse Document Frequency) transforma cada texto em um vetor numérico com base na frequência das palavras, penalizando palavras muito comuns em todos os textos.

Configurações usadas no `TfidfVectorizer`:

| Parâmetro | Valor | Justificativa |
| --- | --- | --- |
| `max_features` | 50.000 | Limita o vocabulário às 50 mil palavras mais relevantes |
| `ngram_range` | (1, 2) | Considera palavras isoladas e pares de palavras |
| `sublinear_tf` | True | Aplica escala logarítmica na frequência do termo |
| `strip_accents` | `unicode` | Remove acentos para normalização |
| `analyzer` | `word` | Analisa em nível de palavras |

O vetorizador foi ajustado somente nos dados de treino (`train_processed.csv`) para evitar vazamento de informação.

## Modelo baseline escolhido

Foram treinados dois modelos para comparação:

### Logistic Regression

- Solver: `lbfgs` com `max_iter=1000`
- Classe pesada: balanceamento automático desativado (dataset já balanceado)

### Linear SVM (LinearSVC)

- Parâmetro C: 1.0
- Máximo de iterações: 2.000

Ambos os modelos foram treinados com os 108.000 exemplos do conjunto de treino.

## Tempo de treinamento (aproximado)

| Modelo | Tempo de treinamento |
| --- | --- |
| Logistic Regression | ~90 segundos |
| LinearSVC | ~30 segundos |

## Modelos salvos

Após o treinamento, os modelos e o vetorizador foram salvos para reutilização:

| Arquivo | Conteúdo |
| --- | --- |
| `models/tfidf_vectorizer.pkl` | Vetorizador TF-IDF ajustado |
| `models/baseline_logreg.pkl` | Modelo Logistic Regression treinado |
| `models/baseline_svm.pkl` | Modelo LinearSVC treinado |

## Conclusão da sprint

Os textos foram vetorizados com TF-IDF e dois modelos baseline foram treinados com sucesso. A próxima etapa é avaliar o desempenho de cada modelo nos conjuntos de validação e teste.
