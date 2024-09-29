# Tutorial de Configuração de Ambiente para Projetos de Data Science

## 1. Clonar um Repositório Git

Para começar, clone o repositório utilizando o comando `git clone`:

[Saiba mais sobre clonar repositórios](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

```bash
git clone <URL_DO_REPOSITORIO>
```

## 2. Criar um Ambiente Virtual no Repositório

A utilização de ambientes virtuais é altamente recomendada em projetos Python. Veja as vantagens no link abaixo:

[Vantagens em Utilizar uma Virtualenv](https://leandrolessa.com.br/dados/4-razoes-para-utilizar-virtualenv-em-projetos-python/).

Atualize o pip e instale o `virtualenv`:

```bash
python3 -m pip install --upgrade pip  # para sistemas Windows
pip install virtualenv
```

Navegue até a pasta do repositório clonado e crie o ambiente virtual:

```bash
cd <PASTA_DO_REPOSITORIO>
virtualenv -p python3 nome_do_ambiente
```

Ative o ambiente virtual:

```bash
cd nome_do_ambiente\Scripts\activate  # Para Windows
source nome_do_ambiente/bin/activate  # Para Linux/Mac
```

## 3. Instalar Bibliotecas para Projeto de Data Science

Agora, com o ambiente virtual ativado, instale as bibliotecas necessárias:

- Para trabalhar com arrays:

```bash
pip install numpy
```

- Para manipulação de dataframes:

```bash
pip install pandas
```

- Para criar gráficos:

```bash
pip install matplotlib
```

- Para criar gráficos mais elaborados:

```bash
pip install seaborn
```

- Para ajustar modelos de aprendizado de máquina:

```bash
pip install -U scikit-learn
```
