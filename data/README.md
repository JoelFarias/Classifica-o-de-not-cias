# Dados

Esta pasta guarda os arquivos do dataset AG News.

## Estrutura

- `raw/`: arquivos originais do dataset.
- `processed/`: arquivos tratados para modelagem.

## Arquivos esperados

Os arquivos originais devem ficar em `data/raw/`:

- `train.csv`
- `test.csv`

Cada arquivo deve ter as colunas:

- `Class Index`
- `Title`
- `Description`

As classes são:

| Código | Classe |
| --- | --- |
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |
