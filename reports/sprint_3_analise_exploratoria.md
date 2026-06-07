# Sprint 3: Analise Exploratoria

## Objetivo

Gerar informacoes visuais e exemplos para entender melhor o dataset AG News antes da modelagem.

## Arquivos gerados

- `reports/figures/class_distribution_train_test.png`
- `reports/figures/word_count_by_class_boxplot.png`
- `reports/figures/word_count_histogram_train.png`
- `reports/sprint_3_distribuicao_classes.csv`
- `reports/sprint_3_tamanho_textos.csv`
- `reports/sprint_3_exemplos_por_classe.md`

## Distribuicao das classes

| class_name   |   train |   test |
|:-------------|--------:|-------:|
| World        |   30000 |   1900 |
| Sports       |   30000 |   1900 |
| Business     |   30000 |   1900 |
| Sci/Tech     |   30000 |   1900 |

O dataset esta balanceado: cada classe possui a mesma quantidade de exemplos no treino e no teste.

## Tamanho dos textos no treino

Os textos foram medidos usando a coluna `text`, formada pela juncao de `Title` e `Description`.

| class_name   |   count |   mean |   median |   min |   max |
|:-------------|--------:|-------:|---------:|------:|------:|
| World        |   30000 |  38.88 |       39 |     9 |   145 |
| Sports       |   30000 |  37.77 |       37 |     4 |   151 |
| Business     |   30000 |  37.54 |       37 |     8 |   134 |
| Sci/Tech     |   30000 |  37.19 |       36 |     8 |   177 |

## Textos mais curtos no treino

| class_name   |   word_count | Title                                |
|:-------------|-------------:|:-------------------------------------|
| Sports       |            4 | Top of 3rd                           |
| Sports       |            5 | Wild re-sign D Schultz               |
| Sports       |            5 | Predators re-sign D Zidlicky         |
| Sports       |            7 | Blues re-sign D Backman, four others |
| Business     |            8 | Stocks to Watch                      |

## Textos mais longos no treino

| class_name   |   word_count | Title                                                       |
|:-------------|-------------:|:------------------------------------------------------------|
| Sci/Tech     |          177 | 2004 US Senate Outlook                                      |
| Sci/Tech     |          171 | Making RSS Scale                                            |
| Sci/Tech     |          170 | Kyoto is Dead - Long Live Pragmatism                        |
| Sci/Tech     |          168 | Baltimore's  quot;Free Books! quot; Charity in Dire Straits |
| Sci/Tech     |          157 | Bush Visits Canada On Fence-Mending Tour                    |

## Possiveis exemplos ambiguos

| observacao                        | class_name   | title                                                          |
|:----------------------------------|:-------------|:---------------------------------------------------------------|
| Business com tecnologia           | Business     | Stocks End Up, But Near Year Lows (Reuters)                    |
| Business com tecnologia           | Business     | Google IPO: Type in 'confusing,' 'secrecy'                     |
| Sci/Tech com empresas             | Sci/Tech     | Oracle Sales Data Seen Being Released (Reuters)                |
| Sci/Tech com empresas             | Sci/Tech     | Company Said to Be Ready to Clone Pets (AP)                    |
| Sports com contexto internacional | Sports       | Phelps, Thorpe Advance in 200 Freestyle (AP)                   |
| Sports com contexto internacional | Sports       | Dreaming done, NBA stars awaken to harsh Olympic reality (AFP) |

## Observacoes

- As classes estao perfeitamente balanceadas, o que facilita a avaliacao dos modelos.
- A maioria dos textos tem tamanho moderado, adequado para modelos tradicionais e modelos de Deep Learning.
- Existem noticias que podem confundir o modelo, principalmente quando misturam termos de negocios, tecnologia, esportes e contexto internacional.
- A coluna `text` ja esta pronta como entrada principal para os proximos passos, pois combina titulo e descricao.

## Conclusao da sprint

A analise exploratoria confirmou que o dataset e bem distribuido e possui textos de tamanho adequado para classificacao. A proxima etapa e a Sprint 4, com limpeza e preparacao dos dados para modelagem.
