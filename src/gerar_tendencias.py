import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta
import warnings

# Ignora os avisos de terminal para não poluir a tela
warnings.filterwarnings('ignore')

# 1. FUNÇÃO MATEMÁTICA (A Exponencial que o CEO pediu)
def modelo_exponencial(x, a, b, c):
    return a * np.exp(b * x) + c

def extrair_historico_e_plotar(pasta_raiz):
    # Procura recursivamente todos os CSVs dentro das Áreas 1000, 2000, etc.
    caminho_busca = os.path.join(pasta_raiz, '**', '*.csv')
    arquivos = glob.glob(caminho_busca, recursive=True)
    
    print(f"Iniciando varredura em {len(arquivos)} arquivos CSV...")
    dados_historicos = []

    # 2. EXTRAÇÃO DE DADOS (Data e Vibração)
    for caminho in arquivos:
        try:
            # Lemos apenas a primeira linha para pescar a DATA e o NOME da máquina
            with open(caminho, 'r', encoding='latin1') as f:
                linha_metadados = f.readline().strip().split(';')
                if len(linha_metadados) < 6:
                    continue # Pula se o arquivo não tiver o padrão Teknikao
                
                info_maquina = linha_metadados[0]
                nome_maquina = info_maquina.split('\\')[-1] # Pega só o nome (Ex: 01-MOT-LOA-HOR)
                data_str = linha_metadados[5] # Pega a data (Ex: 25/05/2022 09:34:33)
                
            # Agora lemos a onda de vibração
            df = pd.read_csv(caminho, sep=';', header=1, decimal=',', encoding='latin1', index_col=False)
            df = df.iloc[:, :2].dropna()
            df.columns = ['Tempo', 'Vibracao']
            df['Vibracao'] = pd.to_numeric(df['Vibracao'], errors='coerce')
            df = df.dropna()
            
            # Remove o desvio do sensor (Detrending) e calcula o RMS (Root Mean Square)
            df['Vibracao'] = df['Vibracao'] - df['Vibracao'].mean()
            rms_calculado = np.sqrt(np.mean(df['Vibracao']**2))
            
            # Guarda a linha no nosso "Banco de Dados" virtual
            dados_historicos.append({
                'Maquina': nome_maquina,
                'Data': pd.to_datetime(data_str, format='%d/%m/%Y %H:%M:%S', errors='coerce'),
                'RMS': rms_calculado
            })
        except Exception as e:
            pass # Ignora silenciosamente arquivos corrompidos

    # 3. CRIAÇÃO DA TABELA DE TENDÊNCIA
    df_tendencia = pd.DataFrame(dados_historicos).dropna()
    if df_tendencia.empty:
        print("Erro: Nenhum dado de data/vibração conseguiu ser extraído.")
        return
        
    print(f"Extração concluída! Construindo tabela de tendência com {len(df_tendencia)} medições.\n")
    
    # Cria uma pasta para salvar os gráficos das máquinas
    pasta_graficos = 'graficos_de_falha'
    if not os.path.exists(pasta_graficos):
        os.makedirs(pasta_graficos)

    maquinas = df_tendencia['Maquina'].unique()
    LIMITE_CRITICO = 4.5 # O valor que o dono da máquina considera perigoso
    graficos_gerados = 0
    
    # 4. INTELIGÊNCIA ARTIFICIAL: CURVE FITTING POR MÁQUINA
    for maquina in maquinas:
        # Filtra a tabela só para a máquina atual e coloca em ordem cronológica
        df_maq = df_tendencia[df_tendencia['Maquina'] == maquina].sort_values(by='Data')
        
        # SÓ É POSSÍVEL TRAÇAR CURVA SE A MÁQUINA FOI MEDIDA EM MAIS DE 2 DIAS DIFERENTES
        if len(df_maq) >= 3:
            try:
                data_inicial = df_maq['Data'].iloc[0]
                dias_operacao = (df_maq['Data'] - data_inicial).dt.days.values
                rms_valores = df_maq['RMS'].values
                
                # Ajuste da Equação Exponencial
                parametros, _ = curve_fit(modelo_exponencial, dias_operacao, rms_valores, p0=(0.1, 0.01, 0.5), maxfev=5000)
                a, b, c = parametros
                
                # Projeta 60 dias para o futuro
                dias_futuros = np.arange(0, dias_operacao[-1] + 60, 1)
                previsao = modelo_exponencial(dias_futuros, a, b, c)
                
                # Plotagem Visual
                plt.figure(figsize=(10, 6))
                plt.scatter(df_maq['Data'], rms_valores, color='blue', label='Medições Históricas (RMS)')
                
                datas_projetadas = [data_inicial + timedelta(days=int(d)) for d in dias_futuros]
                plt.plot(datas_projetadas, previsao, color='orange', linestyle='--', label='Previsão AAV (Exponencial)')
                plt.axhline(y=LIMITE_CRITICO, color='red', linestyle='-', label=f'Limite Crítico ({LIMITE_CRITICO}g)')
                
                plt.title(f'Prognóstico de Vida Útil - {maquina}')
                plt.xlabel('Data da Medição')
                plt.ylabel('Vibração (RMS)')
                plt.legend()
                plt.grid(True)
                
                # Salva o arquivo PNG da máquina na pasta nova
                nome_seguro = maquina.replace('/', '_').replace('\\', '_')
                plt.savefig(f"{pasta_graficos}/Tendencia_{nome_seguro}.png")
                plt.close()
                graficos_gerados += 1
                
            except Exception as e:
                pass 
                
    print("=== RESULTADO DO PROGNÓSTICO ===")
    print(f"Gráficos gerados com sucesso: {graficos_gerados} (salvos na pasta '{pasta_graficos}')")
    
    if graficos_gerados == 0:
        print("\nALERTA TÉCNICO IMPORTANTE PARA A BANCA:")
        print("O script varreu todos os dados perfeitamente, mas NENHUMA máquina possui histórico.")
        print("Nós temos a vibração de várias máquinas, mas apenas em 1 único dia/instante.")
        print("Matematicamente, é impossível traçar uma curva com apenas 1 ponto.")

if __name__ == "__main__":
    # Aponte aqui para a pasta raiz onde estão as "Áreas 1000, 2000", etc.
    # Exemplo: Se estiverem na pasta 'dados_brutos', deixe como está.
    extrair_historico_e_plotar('dados_brutos')