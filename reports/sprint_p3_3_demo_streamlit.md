# Sprint 3 (Pessoa 3 - Bruno): Demonstração

## Objetivo

Criar uma demonstração simples para classificar uma notícia digitada pelo usuário.

## App criado

- `src/app_streamlit.py`

## Como executar

Primeiro, treinar o modelo:

```bash
python src/models/deep_learning_gru.py
```

Depois, abrir a demo:

```bash
streamlit run src/app_streamlit.py
```

## Funcionamento

A interface permite:

- colar ou digitar o texto de uma notícia;
- clicar no botão de classificação;
- visualizar a classe prevista;
- visualizar a confiança da previsão;
- visualizar as probabilidades para as quatro classes.

## Conclusão

A demo cumpre o requisito de apresentar o modelo final de forma simples e usável.
