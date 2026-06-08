# Classificação de Notícias por Tema

**Equipe:** Bruno Pereira, Joel Carolino e Gabriel César

**GitHub:** [JoelFarias/Classifica-o-de-not-cias](https://github.com/JoelFarias/Classifica-o-de-not-cias/tree/main)

**Dataset:** [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

## Descrição

O **AG News Classification Dataset** é uma base de dados de NLP usada para classificação de notícias por tema. Ele foi construído a partir do corpus AG News, selecionando as quatro maiores categorias de notícias do conjunto original.

Cada notícia pertence a uma das quatro classes:

| Classe | Significado |
| --- | --- |
| World | Notícias internacionais, política global, conflitos e eventos mundiais |
| Sports | Esportes, jogos, campeonatos e atletas |
| Business | Economia, empresas, mercado financeiro e negócios |
| Sci/Tech | Ciência, tecnologia, inovação e computação |

A base possui **120.000 amostras de treino** e **7.600 amostras de teste**. Ela é balanceada: cada classe possui **30.000 notícias para treino** e **1.900 notícias para teste**.

Os arquivos principais são:

- `train.csv`
- `test.csv`

Cada linha contém três informações:

- índice da classe;
- título da notícia;
- descrição/resumo da notícia.

As classes são codificadas assim:

| Código | Classe |
| --- | --- |
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

## Tema do Trabalho

**Classificação automática de notícias por tema usando Deep Learning.**

O objetivo é treinar um modelo que leia o título e a descrição de uma notícia e preveja se ela pertence a uma das quatro categorias: `World`, `Sports`, `Business` ou `Sci/Tech`.

## Divisão Sugerida da Equipe

Para a divisão ficar justa, cada integrante deve ficar responsável por uma parte técnica do projeto e também por explicar seus próprios resultados no relatório/apresentação.

### Pessoa 1: Dados, Análise Exploratória e Preparação

Responsável por deixar os dados prontos para os modelos e explicar como a base funciona.

Organização em sprints:

#### Sprint 1: Estrutura do Projeto e Dataset

Status: concluída.

Objetivo: preparar o ambiente inicial e garantir que os dados estejam disponíveis no projeto.

Atividades:

- Criar ou organizar as pastas principais do projeto:
  - `data/`
  - `notebooks/`
  - `src/`
  - `models/`
  - `reports/`
- Baixar o dataset AG News pelo Kaggle.
- Colocar os arquivos `train.csv` e `test.csv` na pasta correta.
- Criar um notebook ou script inicial para carregar os dados.
- Carregar os arquivos usando `pandas`.
- Conferir se os arquivos foram carregados sem erro.

Entregas da sprint:

- Estrutura inicial de pastas criada.
- Dataset salvo no projeto.
- Notebook ou script conseguindo ler `train.csv` e `test.csv`.

#### Sprint 2: Conferência e Entendimento dos Dados

Status: concluída.

Objetivo: entender a estrutura do dataset e verificar se os dados estão corretos.

Atividades:

- Verificar se as colunas estão corretas:
  - classe;
  - título;
  - descrição.
- Conferir a quantidade total de notícias no treino.
- Conferir a quantidade total de notícias no teste.
- Conferir se existem valores vazios.
- Conferir se existem linhas duplicadas.
- Conferir a quantidade de exemplos em cada classe.
- Criar uma tabela mostrando a distribuição das classes.
- Confirmar o significado dos códigos das classes:
  - `1 = World`
  - `2 = Sports`
  - `3 = Business`
  - `4 = Sci/Tech`

Entregas da sprint:

- Tabela com quantidade de exemplos por classe.
- Verificação de valores vazios e duplicados.
- Explicação curta sobre a estrutura do dataset.

#### Sprint 3: Análise Exploratória

Status: concluída.

Objetivo: gerar informações visuais e exemplos para entender melhor a base.

Atividades:

- Criar um gráfico de barras com a quantidade de notícias por classe.
- Mostrar exemplos reais de notícias das classes:
  - World;
  - Sports;
  - Business;
  - Sci/Tech.
- Criar uma nova coluna juntando título e descrição da notícia.
- Calcular o tamanho médio dos textos.
- Verificar textos muito curtos.
- Verificar textos muito longos.
- Anotar possíveis exemplos ambíguos, como:
  - notícias de tecnologia envolvendo empresas;
  - notícias de esporte com contexto internacional;
  - notícias de negócios com termos científicos ou tecnológicos.

Entregas da sprint:

- Gráfico de distribuição das classes.
- Exemplos reais de cada categoria.
- Análise do tamanho dos textos.
- Observações sobre possíveis dificuldades do dataset.

#### Sprint 4: Pré-processamento e Dados Prontos

Status: concluída.

Objetivo: preparar os textos para serem usados pelos modelos da Pessoa 2 e da Pessoa 3.

Atividades:

- Fazer a limpeza inicial dos textos:
  - transformar em minúsculas;
  - remover espaços extras;
  - tratar pontuação quando necessário;
  - remover caracteres estranhos se aparecerem.
- Garantir que a entrada final do modelo seja `título + descrição`.
- Separar quais dados serão usados para treino, validação e teste.
- Salvar os dados preparados ou deixar o código pronto para gerar os dados preparados.
- Documentar quais transformações foram feitas nos textos.
- Entregar para a Pessoa 2 e Pessoa 3 o formato final dos dados.

Entregas da sprint:

- Dados preparados para modelagem.
- Código de pré-processamento funcionando.
- Explicação das transformações feitas nos textos.
- Arquivo ou notebook pronto para ser reutilizado pelos outros integrantes.

Entregáveis:

- Notebook ou script de exploração dos dados.
- Gráfico de distribuição das classes.
- Tabela com quantidade de exemplos por classe.
- Amostras de notícias de cada categoria.
- Explicação do dataset para o relatório.
- Parte da apresentação sobre dataset e pré-processamento.

### Pessoa 2: Baseline, Métricas e Comparação Inicial

Responsável por criar o primeiro modelo funcional do projeto e definir uma base de comparação para o Deep Learning.

Organização em sprints:

#### Sprint 1: Recebimento e Verificação dos Dados

Status: concluída.

Objetivo: receber os dados preparados pela Pessoa 1 e confirmar que estão no formato correto para o treinamento do modelo baseline.

Atividades:

- Receber os arquivos processados da Pessoa 1:
  - `data/processed/train_processed.csv`
  - `data/processed/validation_processed.csv`
  - `data/processed/test_processed.csv`
- Verificar se as colunas `clean_text` e `label` estão presentes.
- Conferir a quantidade de exemplos em cada conjunto.
- Confirmar que não há valores vazios nas colunas de entrada.
- Confirmar o mapeamento das classes:
  - `0 = World`
  - `1 = Sports`
  - `2 = Business`
  - `3 = Sci/Tech`

Entregas da sprint:

- Confirmação do formato e integridade dos dados recebidos.
- Documentação da verificação inicial.

#### Sprint 2: Criação e Treinamento do Baseline

Status: concluída.

Objetivo: transformar os textos em representação numérica com TF-IDF e treinar modelos baseline.

Atividades:

- Criar o script `src/models/baseline_tfidf.py`.
- Configurar o `TfidfVectorizer` com os parâmetros adequados.
- Ajustar o vetorizador apenas nos dados de treino.
- Treinar dois modelos para comparação:
  - Logistic Regression.
  - LinearSVC.
- Salvar o vetorizador e os modelos treinados em `models/`.

Entregas da sprint:

- Script de treinamento funcionando.
- Vetorizador TF-IDF salvo.
- Dois modelos baseline treinados e salvos.

#### Sprint 3: Métricas e Avaliação

Status: concluída.

Objetivo: avaliar o desempenho dos modelos nos conjuntos de validação e teste e calcular as métricas principais.

Atividades:

- Avaliar os modelos nos conjuntos de validação e teste.
- Calcular as métricas por classe:
  - acurácia;
  - precision;
  - recall;
  - F1-score.
- Gerar a matriz de confusão dos dois modelos.
- Identificar quais classes o baseline acerta mais.
- Identificar quais classes o baseline confunde mais.
- Salvar os resultados em `reports/baseline_results.csv`.

Entregas da sprint:

- Métricas completas dos dois modelos.
- Matrizes de confusão.
- Arquivo CSV com resultados para comparação.

#### Sprint 4: Análise de Erros e Resultados Finais

Status: concluída.

Objetivo: identificar padrões nos erros do baseline e documentar os resultados para comparação com o Deep Learning.

Atividades:

- Separar exemplos de erros do modelo.
- Classificar os padrões de confusão mais comuns.
- Explicar por que esses erros ocorrem.
- Documentar as limitações do TF-IDF.
- Registrar as métricas de referência para a Pessoa 3.

Entregas da sprint:

- Análise dos erros com exemplos reais.
- Explicação dos padrões de confusão.
- Métricas de referência documentadas para a Pessoa 3.

Entregáveis:

- Script `src/models/baseline_tfidf.py`.
- Resultados das métricas em `reports/baseline_results.csv`.
- Matrizes de confusão documentadas.
- Análise dos erros com exemplos.
- Explicação do funcionamento do baseline para o relatório.
- Parte da apresentação sobre baseline e primeiros resultados.

### Pessoa 3: Modelo de Deep Learning, Demo e Resultado Final

Responsável por criar o modelo principal com Deep Learning e comparar o resultado final com o baseline.

Atividades:

- Receber os dados preparados pela Pessoa 1.
- Usar as métricas da Pessoa 2 como referência de comparação.
- Escolher o modelo de Deep Learning que será usado:
  - LSTM;
  - GRU;
  - BERT;
  - DistilBERT.
- Caso use LSTM ou GRU:
  - tokenizar os textos;
  - definir o tamanho máximo das sequências;
  - criar camada de embedding;
  - criar camada LSTM ou GRU;
  - adicionar camada Dense;
  - finalizar com Softmax para quatro classes.
- Caso use BERT ou DistilBERT:
  - carregar o tokenizer do modelo;
  - tokenizar os textos no formato esperado;
  - configurar o modelo para classificação com quatro classes;
  - fazer fine-tuning usando os dados de treino.
- Treinar o modelo de Deep Learning.
- Avaliar o modelo nos dados de validação ou teste.
- Calcular as mesmas métricas usadas no baseline:
  - acurácia;
  - precision;
  - recall;
  - F1-score.
- Gerar matriz de confusão do modelo de Deep Learning.
- Comparar os resultados do Deep Learning com o baseline.
- Verificar se o modelo de Deep Learning melhorou ou não.
- Explicar possíveis motivos para melhora ou piora.
- Criar uma demonstração simples com Streamlit ou Gradio:
  - campo para digitar ou colar uma notícia;
  - botão para classificar;
  - resultado com a classe prevista;
  - probabilidade ou confiança da previsão.
- Salvar o modelo final, se possível.

Entregáveis:

- Notebook ou script do modelo de Deep Learning.
- Resultado das métricas do modelo final.
- Matriz de confusão do modelo final.
- Comparação baseline vs Deep Learning.
- Demo simples funcionando.
- Explicação do modelo final para o relatório.
- Parte da apresentação sobre Deep Learning, comparação e demo.

### Responsabilidade Compartilhada

- Todos devem revisar o README final.
- Todos devem participar da apresentação.
- Cada pessoa deve apresentar a parte que desenvolveu.
- As decisões finais do projeto, como escolher o melhor modelo e discutir limitações, devem ser feitas em grupo.

## TODO do Projeto

### 1. Definir escopo do projeto

- [ ] Tema: classificação automática de notícias por tema usando Deep Learning.
- [ ] Dataset: AG News Classification Dataset.
- [ ] Classes: World, Sports, Business, Sci/Tech.
- [ ] Entrada do modelo: título + descrição da notícia.
- [ ] Saída do modelo: uma das quatro categorias.

### 2. Preparar o ambiente

- [ ] Criar notebook no Google Colab ou Jupyter.
- [ ] Instalar bibliotecas necessárias:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `tensorflow` ou `pytorch`
  - `transformers`, caso use BERT ou DistilBERT
- [ ] Baixar o dataset do Kaggle ou carregar via Hugging Face/TensorFlow Datasets.

### 3. Carregar e entender os dados

- [ ] Ler `train.csv` e `test.csv`.
- [ ] Verificar colunas: classe, título e descrição.
- [ ] Conferir quantidade de exemplos por classe.
- [ ] Juntar título + descrição em uma única coluna de texto.
- [ ] Ver exemplos reais de cada categoria.

### 4. Fazer análise exploratória

- [ ] Mostrar distribuição das classes.
- [ ] Calcular tamanho médio dos textos.
- [ ] Ver palavras mais frequentes por classe.
- [ ] Identificar exemplos ambíguos, como notícias de tecnologia empresarial ou esporte internacional.

### 5. Pré-processar os textos

- [ ] Converter texto para minúsculas.
- [ ] Remover caracteres desnecessários.
- [ ] Tratar pontuação, números e espaços extras.
- [ ] Tokenizar os textos.
- [ ] Definir tamanho máximo da sequência.
- [ ] Separar treino, validação e teste.

### 6. Criar modelo baseline

- [ ] Treinar TF-IDF + Logistic Regression ou Naive Bayes.
- [ ] Avaliar acurácia e F1-score.
- [ ] Salvar os resultados para comparação.

### 7. Criar modelo de Deep Learning

Escolher uma das opções abaixo.

#### Opção mais simples

- [ ] Embedding.
- [ ] LSTM ou GRU.
- [ ] Dense.
- [ ] Softmax.

#### Opção mais forte

- [ ] Usar DistilBERT ou BERT.
- [ ] Tokenizar com o tokenizer do modelo.
- [ ] Fine-tunar para quatro classes.
- [ ] Avaliar no conjunto de teste.

### 8. Avaliar os modelos

- [ ] Calcular acurácia.
- [ ] Calcular precision, recall e F1-score.
- [ ] Gerar matriz de confusão.
- [ ] Comparar baseline vs modelo de Deep Learning.
- [ ] Identificar quais classes confundem mais.

### 9. Criar demonstração

- [ ] Fazer uma interface simples no Streamlit ou Gradio.
- [ ] Criar campo para o usuário colar uma notícia.
- [ ] Criar botão para classificar.
- [ ] Mostrar resultado: World, Sports, Business ou Sci/Tech.
- [ ] Mostrar probabilidade/confiança do modelo.

### 10. Montar relatório

- [ ] Introdução ao problema.
- [ ] Descrição do dataset.
- [ ] Explicação das classes.
- [ ] Metodologia.
- [ ] Modelos usados.
- [ ] Resultados.
- [ ] Matriz de confusão.
- [ ] Limitações.
- [ ] Conclusão.

### 11. Montar apresentação

- [ ] Slide 1: tema e integrantes.
- [ ] Slide 2: problema.
- [ ] Slide 3: dataset AG News.
- [ ] Slide 4: pipeline do projeto.
- [ ] Slide 5: modelos usados.
- [ ] Slide 6: resultados.
- [ ] Slide 7: demo.
- [ ] Slide 8: limitações e próximos passos.
