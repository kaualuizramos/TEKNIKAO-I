import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Leitura dos Dados (exatamente como você fez e funcionou)
caminho_do_arquivo = 'COLE_O_CAMINHO_AQUI'
df = pd.read_csv('AmbienteVirtualPython/01-MOT-LOA-HOR.csv', sep=';', header=1, decimal=',')
df = df.iloc[:, :2].dropna()
df.columns = ['Tempo', 'Vibracao']

# 2. Preparação Matemática para a FFT
N = len(df) # Número total de amostras
# Calcula o intervalo de tempo (dt) entre uma medição e outra
dt = df['Tempo'].iloc[1] - df['Tempo'].iloc[0] 

# 3. Aplicação da Transformada de Fourier
# Calcula as amplitudes das frequências
fft_valores = np.fft.fft(df['Vibracao'])
# Calcula os eixos X (quais são as frequências em Hz)
frequencias = np.fft.fftfreq(N, dt)

# Pegamos apenas a metade positiva do gráfico (frequências reais)
frequencias_positivas = frequencias[:N//2]
# Normalizamos e pegamos o valor absoluto da amplitude
amplitudes_positivas = (2.0/N) * np.abs(fft_valores[0:N//2])

# 4. Desenhando o Espectro de Frequências (O gráfico de barras da Teknikao)
plt.figure(figsize=(12, 6))
plt.plot(frequencias_positivas, amplitudes_positivas, color='red')
plt.title('Espectro de Frequências (FFT) - Assinatura da Máquina')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude (g)')
plt.grid(True)

# 5. Salvando o resultado
plt.savefig('grafico_fft.png')
print("Sucesso! A Transformada de Fourier foi calculada.")
print("Abra o arquivo 'grafico_fft.png' para ver os picos de frequência.")