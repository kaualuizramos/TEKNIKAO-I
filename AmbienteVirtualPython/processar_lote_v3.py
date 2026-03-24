import pandas as pd
import numpy as np
import glob
import os

def processar_todos_arquivos(pasta_dados):
    # O segredo está aqui: o '**' e o recursive=True fazem o Python vasculhar TODAS as subpastas!
    caminho_busca = os.path.join(pasta_dados, '**', '*.csv')
    caminhos_arquivos = glob.glob(caminho_busca, recursive=True)
    
    dados_compilados = []
    
    print(f"Buscando na pasta: {pasta_dados}")
    print(f"Encontrados {len(caminhos_arquivos)} arquivos para processar. Aguarde, isso pode levar alguns minutos...")

    for caminho in caminhos_arquivos:
        try:
            # 1. Leitura do arquivo CSV individual
            df = pd.read_csv(caminho, sep=';', header=1, decimal=',', encoding='latin1')
            df = df.iloc[:, :2].dropna()
            df.columns = ['Tempo', 'Vibracao']
            df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')
            df['Vibracao'] = pd.to_numeric(df['Vibracao'], errors='coerce')
            df = df.dropna()
            
            # Se o arquivo estiver vazio, pula para o próximo
            if len(df) == 0:
                continue

            # 2. Hurbilketa (Abordagem 2): Subtrai a média para remover o "desvio" do sensor
            df['Vibracao'] = df['Vibracao'] - df['Vibracao'].mean()

            # 3. Cálculo da FFT (Transformada de Fourier)
            N = len(df)
            dt = df['Tempo'].iloc[1] - df['Tempo'].iloc[0]
            
            # Ignora arquivos onde o tempo esteja corrompido ou zerado
            if dt <= 0:
                continue

            fft_valores = np.fft.fft(df['Vibracao'])
            frequencias = np.fft.fftfreq(N, dt)
            
            amplitudes_positivas = (2.0/N) * np.abs(fft_valores[0:N//2])
            frequencias_positivas = frequencias[:N//2]
            
            # 4. Hurbilketa (Abordagem 1): Ignora a frequência de 0 Hz cortando o array [1:]
            amplitudes_reais = amplitudes_positivas[1:]
            frequencias_reais = frequencias_positivas[1:]
            
            # 5. Extração da Característica (Acha o pico real da falha)
            indice_pico = np.argmax(amplitudes_reais)
            frequencia_pico = frequencias_reais[indice_pico]
            amplitude_maxima = amplitudes_reais[indice_pico]
            
            nome_maquina = os.path.basename(caminho)
            
            # Guarda os resultados da máquina atual
            dados_compilados.append({
                'Maquina': nome_maquina,
                'Frequencia_Pico_Hz': frequencia_pico,
                'Amplitude_Maxima_g': amplitude_maxima
            })
            
        except Exception as e:
            # Ignora arquivos corrompidos silenciosamente para não poluir o terminal
            pass
            
    # 6. Salva tudo num único arquivo final
    if len(dados_compilados) > 0:
        df_final = pd.DataFrame(dados_compilados)
        df_final.to_csv('dataset_treino.csv', index=False)
        print(f"\nProcessamento concluído! O arquivo 'dataset_treino.csv' foi gerado com sucesso e contém {len(dados_compilados)} linhas.")
    else:
        print("\nNenhum dado válido foi encontrado para salvar.")

if __name__ == "__main__":
    # 1. Pega a pasta onde o script está
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    # 2. Volta uma pasta para trás
    pasta_raiz = os.path.dirname(diretorio_script)
    # 3. Entra na pasta de dados extraídos
    pasta_dados = os.path.join(pasta_raiz, 'dados_brutos')
    
    processar_todos_arquivos(pasta_dados)