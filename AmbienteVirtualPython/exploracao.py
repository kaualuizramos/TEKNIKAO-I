import pandas as pd
import matplotlib.pyplot as plt

# 1. Lemos o arquivo. 
# sep=';' separa as colunas. header=1 pula a primeira linha bagunçada. decimal=',' entende os números brasileiros.
df = pd.read_csv('01-MOT-LOA-HOR.csv', sep=';', header=1, decimal=',')

# 2. Pegamos só as duas primeiras colunas (Tempo e Vibração) e removemos linhas vazias
df = df.iloc[:, :2].dropna()
df.columns = ['Tempo', 'Vibracao'] 

# 3. Desenhamos o gráfico
plt.figure(figsize=(10, 5))
plt.plot(df['Tempo'], df['Vibracao'], color='blue')
plt.title('Meu Primeiro Gráfico SDAV')

# 4. Salvamos como imagem
plt.savefig('primeiro_teste.png')
print("Sucesso! Abra o arquivo 'primeiro_teste.png' no canto esquerdo.")