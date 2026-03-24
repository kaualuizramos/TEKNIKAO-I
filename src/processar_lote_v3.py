import pandas as pd
import numpy as np
import glob
import os

def processar_todos_arquivos(pasta_dados):
    caminho_busca = os.path.join(pasta_dados, '**', '*.csv')
    caminhos_arquivos = glob.glob(caminho_busca, recursive=True)
    
    dados_compilados = []
    print(f"Buscando na pasta: {pasta_dados}")
    print(f"Encontrados {len(caminhos_arquivos)} arquivos para processar. Aguarde...")

    for caminho in caminhos_arquivos:
        try:
            # CORREÇÃO DO BUG 1: index_col=False impede que o Pandas embaralhe as colunas
            df = pd.read_csv(caminho, sep=';', header=1, decimal=',', encoding='latin1', index_col=False)
            df = df.iloc[:, :2].dropna()
            df.columns = ['Tempo', 'Vibracao']
            
            df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')
            df['Vibracao'] = pd.to_numeric(df['Vibracao'], errors='coerce')
            df = df.dropna()
            
            if len(df) == 0:
                continue

            # Detrending: centraliza o gráfico no eixo zero
            df['Vibracao'] = df['Vibracao'] - df['Vibracao'].mean()

            N = len(df)
            
            # CORREÇÃO DO BUG 2: O tempo exportado está em Milissegundos!
            dt_ms = df['Tempo'].iloc[1] - df['Tempo'].iloc[0]
            dt_segundos = dt_ms / 1000.0  # Converte para os Segundos reais exigidos pela FFT
            
            if dt_segundos <= 0:
                continue

            # Cálculo Matemático da FFT
            fft_valores = np.fft.fft(df['Vibracao'])
            frequencias = np.fft.fftfreq(N, dt_segundos)
            
            amplitudes_positivas = (2.0/N) * np.abs(fft_valores[0:N//2])
            frequencias_positivas = frequencias[:N//2]
            
            # Ignora a frequência exata de 0 Hz (ruído estático)
            amplitudes_reais = amplitudes_positivas[1:]
            frequencias_reais = frequencias_positivas[1:]
            
            # Encontra o Maior Pico (A Frequência da Falha Mecânica)
            indice_pico = np.argmax(amplitudes_reais)
            frequencia_pico = frequencias_reais[indice_pico]
            amplitude_maxima = amplitudes_reais[indice_pico]
            
            nome_maquina = os.path.basename(caminho)
            
            dados_compilados.append({
                'Maquina': nome_maquina,
                'Frequencia_Pico_Hz': frequencia_pico,
                'Amplitude_Maxima_g': amplitude_maxima
            })
            
        except Exception as e:
            pass
            
    # Salva o arquivo e exibe a contagem
    if len(dados_compilados) > 0:
        df_final = pd.DataFrame(dados_compilados)
        df_final.to_csv('dataset_treino.csv', index=False)
        print(f"\nSucesso! O 'dataset_treino.csv' foi gerado com {len(dados_compilados)} linhas.")
    else:
        print("\nNenhum dado válido foi encontrado para salvar.")

if __name__ == "__main__":
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    pasta_raiz = os.path.dirname(diretorio_script)
    pasta_dados = os.path.join(pasta_raiz, 'dados_brutos')
    
    processar_todos_arquivos(pasta_dados)