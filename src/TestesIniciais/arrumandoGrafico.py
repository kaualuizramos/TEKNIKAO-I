import pandas as pd
import matplotlib.pyplot as plt

caminho_do_arquivo = 'AmbienteVirtualPython/01-MOT-LOA-HOR.csv'

# Lemos o arquivo forçando o ponto e vírgula e a vírgula brasileira
df = pd.read_csv('AmbienteVirtualPython/01-MOT-LOA-HOR.csv', sep=';', header=1, decimal=',')

# Pegamos as duas primeiras colunas e renomeamos
df = df.iloc[:, :2].dropna()
df.columns = ['Tempo', 'Vibracao'] 

# FORÇAMOS a conversão para número decimal (float). Se der erro, ele transforma em NaN (vazio)
df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')
df['Vibracao'] = pd.to_numeric(df['Vibracao'], errors='coerce')

# Removemos qualquer linha que tenha ficado vazia após a conversão
df = df.dropna()

print("Verificação dos dados lidos:")
print(df.head()) # Isso vai mostrar no terminal se os números estão corretos agora!

# Resto do código do gráfico continua igual...
plt.figure(figsize=(10, 5))
plt.plot(df['Tempo'], df['Vibracao'], color='blue')
plt.title('Meu Primeiro Gráfico SDAV - Corrigido')
plt.savefig('primeiro_teste.png')