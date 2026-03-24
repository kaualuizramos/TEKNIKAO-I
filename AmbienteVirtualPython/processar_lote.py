import pandas as pd
import numpy as np
import glob
import os

def processar_todos_arquivos(pasta_dados):
    # Procura todos os arquivos CSV dentro da pasta especificada
    caminhos_arquivos = glob.glob(os.path.join(pasta_dados, '*.csv'))
    
    dados_compilados = []
    
    print(f"Encontrados {len(caminhos_arquivos)} arquivos para processar...")

    for caminho in caminhos_arquivos:
        try:
            # 1. Leitura do arquivo individual
            df = pd.read_csv(caminho, sep=';', header=1, decimal=',')
            df = df.iloc[:, :2].dropna()
            df.columns = ['Tempo', 'Vibracao']
            df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')
            df['Vibracao'] = pd.to_numeric(df['Vibracao'], errors='coerce')
            df = df.dropna()
            
            # Se o arquivo estiver vazio, pula para o próximo
            if len(df) == 0:
                continue

            # 2. Cálculo rápido da FFT
            N = len(df)
            dt = df['Tempo'].iloc[1] - df['Tempo'].iloc[0]
            fft_valores = np.fft.fft(df['Vibracao'])
            frequencias = np.fft.fftfreq(N, dt)
            
            amplitudes_positivas = (2.0/N) * np.abs(fft_valores[0:N//2])
            frequencias_positivas = frequencias[:N//2]
            
            # 3. Extração da Característica (Feature Engineering)
            # Encontra qual é a amplitude máxima e em qual frequência ela ocorreu
            indice_pico = np.argmax(amplitudes_positivas)
            frequencia_pico = frequencias_positivas[indice_pico]
            amplitude_maxima = amplitudes_positivas[indice_pico]
            
            # Pega o nome do arquivo para saber de qual máquina veio
            nome_maquina = os.path.basename(caminho)
            
            # Guarda os resultados da máquina atual
            dados_compilados.append({
                'Maquina': nome_maquina,
                'Frequencia_Pico_Hz': frequencia_pico,
                'Amplitude_Maxima_g': amplitude_maxima,
                # Podemos adicionar depois: média de vibração, RMS, etc.
            })
            
        except Exception as e:
            print(f"Erro ao processar o arquivo {caminho}: {e}")
            
    # 4. Salva tudo num único arquivo estruturado
    df_final = pd.DataFrame(dados_compilados)
    df_final.to_csv('dataset_treino.csv', index=False)
    print("Processamento concluído! O arquivo 'dataset_treino.csv' foi gerado.")

if __name__ == "__main__":
    # Descobre automaticamente a pasta onde este script processar_lote.py está salvo
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    
    # Roda a função mandando ela procurar os CSVs nesta mesma pasta
    processar_todos_arquivos(pasta_atual)