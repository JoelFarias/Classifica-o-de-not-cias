Descrição do dataset AG News
O AG News Classification Dataset é uma base de dados de NLP para classificação de notícias por tema. Ele foi construído a partir do corpus AG News, selecionando as quatro maiores categorias de notícias do conjunto original. Cada notícia pertence a uma de quatro classes: World, Sports, Business ou Sci/Tech. 
Blog do Hvitfeldt · 1
A base possui 120.000 amostras de treino e 7.600 amostras de teste. Ela é balanceada: cada classe tem 30.000 notícias para treino e 1.900 notícias para teste. 
Kaggle · 1
Os arquivos principais são train.csv e test.csv. Cada linha contém três informações: índice da classe, título da notícia e descrição/resumo da notícia. As classes são codificadas assim: 1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech. 
Hugging Face
Tema do trabalho
Classificação automática de notícias por tema usando Deep Learning
O objetivo é treinar um modelo que leia o título e a descrição de uma notícia e preveja se ela pertence a uma destas categorias:
Classe
Significado
World
Notícias internacionais, política global, conflitos, eventos mundiais
Sports
Esportes, jogos, campeonatos, atletas
Business
Economia, empresas, mercado financeiro, negócios
Sci/Tech
Ciência, tecnologia, inovação, computação
