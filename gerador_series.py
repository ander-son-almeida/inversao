import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_time_series(num_periods, num_series, 
                         crossover_probability, 
                         crossover_periods, random_seed=None):
    
    # Define a semente para reprodutibilidade
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Inicializa as séries temporais
    series = np.zeros((num_series, num_periods))
    
    # Define um offset inicial para evitar cruzamento prematuro
    offset = np.linspace(0, 0.8, num_series)  # Distribui as séries entre 0 e 0.8
    series[:, 0] = offset  # Aplica o offset inicial
    
    # Gera as séries temporais com algum grau de volatilidade
    for i in range(1, num_periods):
        volatility = np.random.normal(0, 0.03, num_series)  # Volatilidade aleatória (reduzida para evitar cruzamentos)
        series[:, i] = series[:, i-1] + volatility
        series[:, i] = np.clip(series[:, i], 0, 1)  # Mantém os valores entre 0 e 1
    
    # Verifica se há cruzamento nos últimos `crossover_periods` 
    for i in range(num_series - 1):
        for j in range(i + 1, num_series):
            if np.random.rand() < crossover_probability:
                # Inverte as séries a partir do período de cruzamento
                crossover_point = num_periods - np.random.randint(1, crossover_periods + 1)
                series[i, crossover_point:], series[j, crossover_point:] = series[j, crossover_point:], series[i, crossover_point:].copy()
    
    return series

def create_dataframe(series, start_date):
    # Cria um DataFrame com as séries temporais
    num_series, num_periods = series.shape
    
    # Gera as datas (primeiro dia de cada mês)
    dates = pd.date_range(start=start_date, periods=num_periods, freq='MS')  # MS = Month Start
    
    # Cria o DataFrame
    df = pd.DataFrame(series.T, index=dates)
    df.columns = [f'Folha {i+1}' for i in range(num_series)]
    
    return df

def plot_time_series(df):
    
    df.plot(marker='.')
    plt.ylabel('PD ou LGD')
    plt.grid()
    plt.title('Simulação de Folhas (Inversão)')
    plt.show()

# Entradas 
num_periods = 48
num_series = 4
crossover_probability = 0.5
crossover_periods = 6
random_seed = 5
start_date = "2024-01-01"

# Gera as séries temporais
series = generate_time_series(num_periods, num_series, 
                              crossover_probability, crossover_periods, 
                              random_seed)

# Cria o DataFrame com as datas no formato ano-mês-dia
df = create_dataframe(series, start_date)

plot_time_series(df)



# detector de inversao
arrays = [df[col].to_numpy() for col in df.columns]


# Número de arrays
n = len(arrays)

# Criar um DataFrame para armazenar as porcentagens
result_df = pd.DataFrame(index=df.columns, columns=df.columns)

# calcular a porcentagem de diferenças
def calc_diff(arr1, arr2):
    diff = arr1 - arr2
    result = np.where(diff > 0, 1, -1) # valores maior que 0 é marcado com 1 caso contrario -1
    first_value = result[0] # para determinar inversao, vou tomar com referencia o primeiro valor da serie
    different_values = len(result[result != first_value]) # contando o numero de elementos diferentes do primeiro valor da serie
    percentage = (different_values / len(result)) # com a qtd de elementos determinado, verifico qual a parcela que esses elementos são do total da serie
    return percentage

# comparando folha por folha
for i in range(n):
    for j in range(i + 1, n):  # Evita comparações redundantes
        percentage = calc_diff(arrays[i], arrays[j])
        result_df.iloc[i, j] = percentage
        result_df.iloc[j, i] = percentage  # Simetria: reutiliza o resultado

# Preencher a diagonal principal com 0 (comparação de uma array com ela mesma)
np.fill_diagonal(result_df.values, 0)


min_meses_inv = 3 # quantidade minima de meses para "marcar" como uma inversao
# detalhe que estamos olhando ao longo de toda vida da serie. Se ocorrer inversao
# por mais de 4 meses em um passado distante, ainda assim vai ser marcado como inversao

def mark_values(val):
    tolerancia = min_meses_inv/num_periods
    print(tolerancia)
    return 'Sim' if val > tolerancia else 'Não'

# Aplicar a função a cada elemento do DataFrame
final_df = result_df.applymap(mark_values)
print(final_df)





































