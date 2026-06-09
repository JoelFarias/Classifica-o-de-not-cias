# Correção da Entrega DistilBERT do Bruno

## Objetivo

A pasta `ProjetoNoticias/` enviada pelo Bruno tinha uma proposta de modelo DistilBERT, mas os arquivos não estavam prontos para uso no repositório principal.

## Problemas encontrados

- Arquivos principais com nome duplicado, como `app (1).py`.
- Arquivos originais `app.py` e `deep_learning_distilbert.py` vazios.
- Dependência de `datasets` e `accelerate`, que não estavam instaladas no ambiente.
- Caminhos incorretos para os dados:
  - usava `data/processed/train.csv`;
  - usava `data/processed/test.csv`.
- O projeto usa:
  - `data/processed/train_processed.csv`;
  - `data/processed/validation_processed.csv`;
  - `data/processed/test_processed.csv`.
- O app tentava carregar `models/distilbert_news_model`, mas esse modelo ainda não existia.
- O script usava `pandas.DataFrame.sample(..., stratify=...)`, parâmetro que não existe no pandas.

## Correção feita

Foram criados arquivos corrigidos no padrão do projeto:

| Arquivo | Função |
| --- | --- |
| `src/models/deep_learning_distilbert.py` | Treino e avaliação do DistilBERT |
| `src/app_distilbert.py` | Demo opcional para o modelo DistilBERT |

## Diferenças da versão corrigida

- Usa os arquivos processados corretos.
- Não depende de `datasets` nem de `accelerate`.
- Usa `torch.utils.data.Dataset` e `DataLoader`.
- Faz amostragem estratificada manualmente.
- Salva métricas em `reports/distilbert_results.csv`.
- Salva matriz de confusão em `reports/distilbert_confusion_matrix.csv`.
- Salva o modelo em `models/distilbert_news_model`.

## Como executar

```bash
python src/models/deep_learning_distilbert.py
streamlit run src/app_distilbert.py
```

## Validação

O fluxo foi testado com uma amostra mínima de 40 exemplos para confirmar que:

- o script carrega os dados processados corretos;
- o tokenizer e o modelo DistilBERT são carregados;
- o treinamento executa;
- a avaliação executa;
- o modelo e os relatórios são gerados.

Os artefatos dessa execução mínima foram removidos porque os resultados não representam uma avaliação real do modelo.

## Observação

O modelo GRU continua sendo a demo principal já treinada e validada. A versão DistilBERT foi integrada como alternativa corrigida da proposta enviada pelo Bruno.
