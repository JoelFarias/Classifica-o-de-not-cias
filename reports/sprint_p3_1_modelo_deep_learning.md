# Sprint 1 (Pessoa 3 - Bruno): Modelo de Deep Learning

## Objetivo

Criar um modelo de Deep Learning para classificar notícias do dataset AG News e comparar seu desempenho com o baseline da Pessoa 2.

## Modelo escolhido

Foi escolhido um modelo **GRU bidirecional** implementado em PyTorch.

Essa opção foi usada porque:

- é mais leve que BERT ou DistilBERT;
- treina localmente em menos tempo;
- ainda é um modelo de Deep Learning real;
- consegue processar sequência de palavras, diferente do TF-IDF.

## Configuração final

| Item | Valor |
| --- | ---: |
| Épocas | 10 |
| Amostras de treino | 40.000 |
| Vocabulário | 30.000 tokens |
| Embedding | 128 |
| Hidden size | 96 |
| Dropout | 0,3 |
| Dispositivo | GPU CUDA |

## Script criado

- `src/models/deep_learning_gru.py`

O script faz:

- carregamento dos dados processados;
- tokenização dos textos;
- criação do vocabulário;
- padding das sequências;
- treinamento do modelo GRU;
- avaliação em validação e teste;
- geração de métricas;
- geração da matriz de confusão;
- salvamento do modelo final.

## Dados usados

| Conjunto | Uso |
| --- | --- |
| `data/processed/train_processed.csv` | Treino |
| `data/processed/validation_processed.csv` | Validação |
| `data/processed/test_processed.csv` | Teste final |

## Conclusão

A estrutura da parte de Deep Learning foi implementada e está pronta para execução e comparação com o baseline.
