import pandas as pd

## COLOCAR TRATATIVA DE ERRO NAS FUNÇÕES
## = na declaração é valor default
def count_duplicates(dataframe: pd.DataFrame = pd.DataFrame({'C1':[0,1,2,3],'C2':[4,5,6,7]})):
    
    '''
    Retorna a quantidade de dados duplicados em um dataframe.

    Parâmetros
    ____________
    dataframe: pd.DataFrame
        DataFrame que será analisado

    Return
    ___________
    int
        A quantidade de registros duplicados

    '''
    try:
        qtd = dataframe.duplicated().sum()
        print(f'Foram encontrados {qtd} de registros duplicados')
        return qtd
    except Exception as e:
        print(f'Não foi possível encontrar a quantidade de dados duplicados. Erro: {e}')

def show_data (dataframe: pd.DataFrame):

    '''
    Retorna a quantidade de dados duplicados em um dataframe.

    Parâmetros
    ____________
    dataframe: pd.DataFrame
        DataFrame que será analisado

    Return
    ___________
    int
        A quantidade de registros duplicados

    '''

    return dataframe.sample(10)

def check_duplicates(dataframe: pd.DataFrame):

    '''
    Retorna a quantidade de dados duplicados em um dataframe.

    Parâmetros
    ____________
    dataframe: pd.DataFrame
        DataFrame que será analisado

    Return
    ___________
    int
        A quantidade de registros duplicados

    '''

    print('Lista com os dados duplicados')
    return dataframe.duplicated()


def drop_duplicates(dataframe: pd.DataFrame):

    '''
    Retorna a quantidade de dados duplicados em um dataframe.

    Parâmetros
    ____________
    dataframe: pd.DataFrame
        DataFrame que será analisado

    Return
    ___________
    int
        A quantidade de registros duplicados

    '''

    return dataframe.drop_duplicates()