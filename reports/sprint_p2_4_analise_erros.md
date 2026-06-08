# Sprint 4 (Pessoa 2): Análise de Erros e Resultados Finais

## Objetivo

Identificar padrões nos erros do modelo baseline, explicar por que esses erros ocorrem e registrar os resultados para comparação com o modelo de Deep Learning da Pessoa 3.

## Exemplos de erros do LinearSVC

Os exemplos abaixo foram classificados incorretamente pelo modelo.

### Erro: Business classificado como Sci/Tech

> **Texto:** "oracle faces antitrust probe europe european regulators investigating oracle's acquisition business practices software market"
>
> **Classe real:** Business
> **Classe prevista:** Sci/Tech

**Por que errou:** A notícia envolve uma empresa de tecnologia (Oracle) e usa termos do universo tech. O modelo associou as palavras ao contexto tecnológico em vez do contexto empresarial.

---

### Erro: Sci/Tech classificado como Business

> **Texto:** "tech stocks slide nasdaq falls percent chip makers earnings disappoint investors quarterly results"
>
> **Classe real:** Sci/Tech
> **Classe prevista:** Business

**Por que errou:** O texto fala de bolsa de valores, resultados trimestrais e investidores — vocabulário tipicamente associado a Business. O contexto tecnológico ficou em segundo plano.

---

### Erro: World classificado como Business

> **Texto:** "imf warns global economic slowdown rising debt emerging markets central banks interest rates"
>
> **Classe real:** World
> **Classe prevista:** Business

**Por que errou:** O texto usa vocabulário financeiro pesado (FMI, taxa de juros, mercados emergentes), que o modelo associou à categoria Business.

---

### Erro: Business classificado como World

> **Texto:** "china us trade war tariffs escalate diplomatic tensions beijing washington negotiations stall"
>
> **Classe real:** Business
> **Classe prevista:** World

**Por que errou:** O contexto de tensão diplomática entre países levou o modelo a classificar como World, ignorando o foco na guerra comercial.

## Padrões de erro identificados

| Confusão | Motivo principal |
| --- | --- |
| Business → Sci/Tech | Notícias sobre empresas de tecnologia com vocabulário técnico |
| Sci/Tech → Business | Notícias sobre resultados financeiros do setor tech |
| World → Business | Notícias sobre economia global com vocabulário financeiro |
| Business → World | Notícias de comércio internacional com contexto diplomático |

## Limitações do baseline TF-IDF

- O TF-IDF não captura o contexto nem a ordem das palavras. Uma palavra como "oracle" pesa igual independentemente de estar numa notícia sobre banco de dados ou sobre religião.
- Pares de palavras (bigramas) ajudam, mas ainda são insuficientes para disambiguar textos que transitam entre dois temas.
- O modelo não entende relações semânticas entre palavras. Para ele, "receita" (resultado financeiro) e "receita" (culinária) são a mesma coisa.

## Resultados salvos para comparação

Os resultados foram salvos para a Pessoa 3 usar como referência:

| Arquivo | Conteúdo |
| --- | --- |
| `reports/baseline_results.csv` | Métricas por classe dos dois modelos |
| `models/tfidf_vectorizer.pkl` | Vetorizador treinado |
| `models/baseline_logreg.pkl` | Modelo Logistic Regression |
| `models/baseline_svm.pkl` | Modelo LinearSVC (melhor baseline) |

## Referência para a Pessoa 3

O modelo de Deep Learning deverá ser comparado com o LinearSVC, que foi o melhor baseline:

| Métrica de referência (LinearSVC) | Valor |
| --- | ---: |
| Acurácia | 91,4% |
| F1 médio | 0,91 |
| F1 Sports | 0,97 |
| F1 Business | 0,88 |
| F1 Sci/Tech | 0,89 |
| F1 World | 0,91 |

## Conclusão da sprint

O baseline foi concluído e os resultados estão documentados. A principal dificuldade identificada foi a separação entre Business e Sci/Tech, que compartilham vocabulário em notícias sobre empresas de tecnologia. Espera-se que o modelo de Deep Learning supere esses resultados ao capturar contexto e semântica que o TF-IDF não consegue representar.
