# Repositório para o processamento de Dados Gravimétricos e Magnéticos

Este repositório contém uma coleção de ferramentas e recursos para manipulação de dados gravimétricos e magnéticos, implementados em Python. Utilizando as bibliotecas `numpy`, `scipy`, `matplotlib`, `pyvista` e `harmonica`, onde se tem uma ampla gama de funcionalidades para processamento, análise e visualização desses tipos de dados geofísicos.

## Pré-requisitos

Antes de utilizar este repositório, certifique-se de ter as seguintes bibliotecas Python instaladas:

- [NumPy](https://numpy.org): Biblioteca para cálculos numéricos eficientes.
- [SciPy](https://scipy.org): Biblioteca para funções científicas e de engenharia.
- [Matplotlib](https://matplotlib.org): Biblioteca para visualização de dados em Python.
- [Jupyter Notebook](https://jupyter.org): Ambiente de desenvolvimento interativo para escrever e executar código Python em blocos.
- [harmonica](https://www.fatiando.org/harmonica/latest/): Biblioteca Python para modelagem direta, inversão e processamento de dados gravitacionais e magnéticos.

## Conteúdo do Repositório

O repositório é organizado da seguinte forma:

- [Anomalia campo total](1_Anomalia_campo_total.ipynb): Este Jupyter notebook ilustra como calcular os componentes do campo magnético e a TFA (Total Field Anomaly), produzidos por prismas retangulares homogêneos. O campo magnético e a TFA são grandezas importantes na área da geofísica e são utilizados para entender as propriedades magnéticas de diferentes formações geológicas.

- [Disturbio gravidade](2_Disturbio_gravidade.ipynb): Este Jupyter Notebook ilustra como calcular o distúrbio de gravidade produzida por prismas retangulares homogêneos. Nesta abordagem, presume-se que o distúrbio de gravidade possa ser aproximado pela componente vertical da atração gravitacional gerada pelos prismas. O distúrbio de gravidade é uma medida das variações na aceleração da gravidade em um determinado local, em relação a um valor de referência. Essas variações podem ser causadas por diferenças na distribuição de massa em subsuperfície, como a presença de prismas retangulares homogêneos.

- [RTP Fourier.ipynb](3_RTP_Fourier.ipynb): Este Jupyter Notebook ilustra como calcular a RTP (Redução ao Polo) usando a Transformada de Fourier das anomalias de campo total produzidas por prismas retangulares homogêneos. A RTP é uma técnica utilizada na área de geofísica para corrigir as anomalias magnéticas medidas em campo para sua localização hipotética no polo magnético da Terra. Essa técnica é particularmente útil para eliminar as distorções causadas pela inclinação magnética e para obter uma imagem mais clara das variações magnéticas de uma área.

- [RTP Camada equivalente.ipynb](4_RTP_Camada_equivalente.ipynb): Este Jupyter Notebook ilustra como calcular a RTP (Redução ao Polo) utilizando a técnica da camada equivalente a partir de anomalias de campo total produzidas por prismas retangulares homogêneos. A RTP é uma técnica amplamente utilizada na área de geofísica para corrigir as anomalias magnéticas medidas em campo. Essa correção é necessária para remover os efeitos indesejados da inclinação magnética e obter uma representação mais precisa das variações magnéticas presentes em uma área de interesse. A técnica da camada equivalente consiste em criar uma camada fictícia de fontes magnéticas equivalentes que reproduz as anomalias de campo total medidas em campo. Essa camada é composta por uma malha de prismas retangulares homogêneos com magnetizações equivalentes.

- [Processamento dados gravimetricos](5_Processamento_dados_gravimetricos.ipynb): Neste tutorial, faremos um tour pelo Harmonica, uma biblioteca Python para modelagem direta, inversão e processamento de dados gravitacionais, com foco no fluxo de trabalho de processamento para produzir uma grade regular do _distúrbio de gravidade Bouguer_.

- [Processamento dados magneticos.ipynb](6_Processamento_dados_magneticos.ipynb): Neste tutorial, vamos explorar como ler e interpolar dados de um levantamento de aerolevantamento na Região de Carajás, Pará, Brasil. A Região de Carajás, conhecida por sua riqueza mineral, é um importante local de exploração de minério de ferro, cobre, ouro e outros minerais. Levantamentos aerogeofísicos são frequentemente realizados nessa região para mapear e investigar a distribuição desses recursos minerais.

- [codes](/codes): Pasta com o código-fonte em Python com implementações abrangentes dos algoritmos de processamento de dados magnéticos e gravimétricos. 

- [figuras](/figuras): Local de armazenamento e salvalmento de figuras.

- [slides](/slides): Conteúdo teorico sobre os assuntos abordados nos notebooks.

- [Dados](/dados): Dados de input e output usados nos notebooks.

## Como utilizar este repositório

1. Certifique-se de ter todas as bibliotecas mencionadas acima instaladas em seu ambiente Python.

2. Clone este repositório em sua máquina local usando o comando `git clone https://github.com/raimundo1994/tutorial-gravmag.git`.

3. Acesse o diretório do repositório clonado.

4. Utilize os notebooks Jupyter fornecidos para explorar os exemplos de uso e aplicar os algoritmos de processamento de dados potenciais em seus próprios dados.

## Contribuições

Se você tiver ideias, sugestões ou melhorias para este repositório, sinta-se à vontade para contribuir. Você pode enviar solicitações de pull com suas contribuições ou abrir uma issue para discutir novas funcionalidades, correções de bugs ou outros assuntos relacionados à manipulação de dados gravimétricos e magnéticos.

