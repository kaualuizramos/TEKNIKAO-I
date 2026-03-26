import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# CONFIGURAÇÕES
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.txt")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
JANELA = 15
PASSOS_FUTURO = 120 # Projeta 4 meses para a frente

def carregar_e_limpar_dados():
    print("📥 Carregando as 35.000 linhas do Dataset...")
    blocos = []
    bloco_atual = []
    
    with open(DATA_PATH, 'r', encoding='latin1') as f:
        for linha in f:
            linha = linha.strip()
            if not linha or len(linha) == 1:
                if len(bloco_atual) > 5: # Só guarda blocos com algum histórico
                    blocos.append(bloco_atual)
                bloco_atual = []
                continue
                
            partes = linha.split()
            if len(partes) >= 2 and '/' in partes[0]:
                try:
                    data_str = partes[0]
                    valor = float(partes[1].replace(',', '.'))
                    bloco_atual.append({'Data': data_str, 'Vibracao': valor})
                except: pass
    if len(bloco_atual) > 5: blocos.append(bloco_atual)
    
    print(f"✅ Encontradas {len(blocos)} sequências de vida útil de máquinas.")
    
    # Processa e interpola todos os blocos para criar dias contínuos
    blocos_processados = []
    for b in blocos:
        df = pd.DataFrame(b)
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna().sort_values('Data')
        df = df.groupby('Data')['Vibracao'].mean().reset_index()
        df = df.set_index('Data').resample('D').interpolate(method='linear').reset_index()
        blocos_processados.append(df['Vibracao'].values)
        
    return blocos_processados

def criar_features_globais(blocos_processados, janela):
    X, y = [], []
    for valores in blocos_processados:
        if len(valores) <= janela: continue
            
        for i in range(len(valores) - janela):
            seq = valores[i:i+janela]
            valor_futuro = valores[i+janela]
            crescimento = valor_futuro - seq[-1]
            
            # O SEGREDO DO PHM: Se a vibração caiu, foi manutenção! Ignoramos.
            if crescimento < -0.05:
                continue
                
            features = list(seq) + [np.mean(seq), np.std(seq), np.max(seq), seq[-1] - seq[0]]
            X.append(features)
            y.append(crescimento) # Treinamos a IA para prever O QUANTO VAI PIORAR
            
    return np.array(X), np.array(y)

def projetar_futuro(modelo, valores_recentes, janela, passos):
    entrada = valores_recentes[-janela:].copy()
    previsoes = []
    
    for _ in range(passos):
        features = list(entrada) + [np.mean(entrada), np.std(entrada), np.max(entrada), entrada[-1] - entrada[0]]
        
        # A IA prevê a taxa de crescimento diária
        delta_previsto = modelo.predict(np.array(features).reshape(1, -1))[0]
        
        # A máquina não se repara sozinha: forçamos um crescimento mínimo se a IA tentar descer
        if delta_previsto < 0: delta_previsto = 0.01 
            
        novo_valor = entrada[-1] + delta_previsto
        previsoes.append(novo_valor)
        entrada = np.append(entrada[1:], novo_valor)
        
    return np.array(previsoes)

def main():
    blocos = carregar_e_limpar_dados()
    X_treino, y_treino = criar_features_globais(blocos, JANELA)
    
    print(f"🧠 Treinando o XGBoost (O Cérebro Global) com {len(X_treino)} exemplos diários...")
    modelo = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    modelo.fit(X_treino, y_treino)
    
    print("🔮 Gerando previsões INDIVIDUAIS para as máquinas com maior histórico...")
    
    # Ordena os blocos de máquinas do maior histórico para o menor
    blocos_ordenados = sorted(blocos, key=len, reverse=True)
    
    # Seleciona apenas as 10 máquinas com mais dados para criar os gráficos de vitrine
    top_maquinas = blocos_ordenados[:10]
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    for indice, maquina in enumerate(top_maquinas):
        # A Inteligência Artificial (treinada com todas) projeta o futuro DESTA máquina específica
        previsoes_futuras = projetar_futuro(modelo, maquina, JANELA, PASSOS_FUTURO)
        
        # ---------------- PLOTAGEM DO GRÁFICO INDIVIDUAL ----------------
        plt.figure(figsize=(12,6))
        
        # Desenha o passado
        plt.plot(maquina, label="Histórico Real (RMS)", color='#1f77b4', linewidth=2)
        
        # Desenha o futuro
        eixo_x_futuro = range(len(maquina), len(maquina) + PASSOS_FUTURO)
        plt.plot(eixo_x_futuro, previsoes_futuras, label="Projeção XGBoost", color='#ff7f0e', linestyle='--', linewidth=2.5)
        
        # Linha de perigo 
        plt.axhline(y=20, color='red', linestyle=':', label='Limite Crítico')
        
        plt.title(f"Prognóstico XGBoost - Componente {indice + 1}", fontsize=14, pad=15)
        plt.xlabel("Dias de Operação")
        plt.ylabel("Nível de Vibração (RMS)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        caminho_grafico = os.path.join(PLOTS_DIR, f"previsao_xgboost_maq_{indice + 1}.png")
        plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
        plt.close() # Fecha a imagem da memória para não travar o PC
        
    print(f"\n🎉 SUCESSO! 10 Gráficos individuais foram salvos na pasta: {PLOTS_DIR}")

if __name__ == "__main__":
    main()