# Tutorial de Configuração de Ambiente para Projetos de Data Science para Windows

## 1. Criar um Ambiente Virtual

A utilização de ambientes virtuais é altamente recomendada em projetos Python. Veja as vantagens no link abaixo:

[Vantagens em Utilizar uma Virtualenv](https://leandrolessa.com.br/dados/4-razoes-para-utilizar-virtualenv-em-projetos-python/).

Atualize o pip e instale o `virtualenv`:

```bash
python3 -m pip install --upgrade pip  # para sistemas Windows
pip install virtualenv
cd <PASTA_DO_AMBIENTEVIRTUAL>
virtualenv -p python3 nome_do_ambiente
```

Ative o ambiente virtual:

```bash
cd nome_do_ambiente\Scripts\activate  # Para Windows
```

## 2. Instalar Bibliotecas para Projeto de Data Science

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

## 3. Clonar um Repositório Git

Para começar, clone o repositório utilizando o comando `git clone`:

[Saiba mais sobre clonar repositórios](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

```bash
git clone <URL_DO_REPOSITORIO>
```

## 4. Subir Arquivos de Trabalho para o Repositório

Agora que você configurou seu ambiente e clonou o repositório, siga os passos abaixo para subir seus arquivos de trabalho para o repositório no GitHub.

### 4.1. Verificar Arquivos Modificados

Verifique os arquivos que foram modificados ou adicionados desde o último commit:

```bash
git status
```

### 4.2. Adicionar Arquivos ao Commit

Adicione os arquivos modificados ou novos para serem incluídos no próximo commit:

```bash
git add .
```

### 4.3. Fazer o Commit das Mudanças

Após adicionar os arquivos, faça o commit com uma mensagem explicativa:

```bash
git commit -m "Mensagem descrevendo as mudanças"
```

### 4.4. Enviar as Mudanças para o Repositório Remoto

Finalmente, envie suas alterações para o repositório remoto:

```bash
git push origin main
```

Caso esteja utilizando um branch diferente do `main`, substitua o nome `main` pelo nome do seu branch.
