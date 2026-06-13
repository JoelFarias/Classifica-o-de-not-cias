# Sprint 4 (Pessoa 3 - Bruno): Resultado Final

## Objetivo

Registrar a entrega final da parte de Deep Learning, incluindo modelo, avaliação, comparação e demonstração.

## Entregas

| Entrega planejada | Status |
| --- | --- |
| Script do modelo de Deep Learning | Concluído |
| Métricas do modelo final | Concluído |
| Matriz de confusão | Concluído |
| Comparação com baseline | Concluído |
| Demo simples funcionando | Criada com Streamlit |
| Explicação do modelo final | Documentada |

## Arquivos principais

| Arquivo | Função |
| --- | --- |
| `src/models/deep_learning_gru.py` | Treino, avaliação e salvamento do modelo |
| `src/app_streamlit.py` | Demo interativa |
| `reports/deep_learning_results.csv` | Métricas |
| `reports/deep_learning_confusion_matrix.csv` | Matriz de confusão |
| `reports/figures/deep_learning_confusion_matrix.png` | Figura da matriz de confusão |

## Resultado final

| Modelo | Acurácia | F1 médio |
| --- | ---: | ---: |
| LinearSVC baseline | cerca de 91% a 92% | cerca de 0,91 a 0,92 |
| GRU Deep Learning | 87,57% | 0,8758 |

O modelo de Deep Learning foi treinado com GPU CUDA, 10 épocas e 40.000 amostras de treino. Ele ficou abaixo do baseline, mas entregou uma comparação válida e melhorou em relação à execução anterior. Para melhorar o resultado, a próxima tentativa poderia usar o dataset completo ou um modelo Transformer como DistilBERT.

## Conclusão

A parte da Pessoa 3 foi estruturada para completar o projeto: agora existe um modelo de Deep Learning, uma forma de avaliar os resultados, uma comparação com o baseline e uma demo para testar notícias manualmente.
