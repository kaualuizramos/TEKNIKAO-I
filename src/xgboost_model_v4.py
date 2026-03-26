import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# CONFIG

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.txt")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

JANELA = 15
PASSOS_FUTURO = 120

# SUAVIZAÇÃO (NÃO POLINOMIAL)

def suavizar_curva(dados, alpha=0.3):
    suavizado = [dados[0]]
    for i in range(1, len(dados)):
        suavizado.append(alpha * dados[i] + (1 - alpha) * suavizado[-1])
    return np.array(suavizado)

# DETECÇÃO DE FALHA

def detectar_falha(previsoes, limite):
    for i, v in enumerate(previsoes):
        if v > limite:
            return i
    return None

# CARREGAR DADOS

def carregar_e_limpar_dados():
    print("📥 Carregando dataset...")

    blocos = []
    bloco_atual = []

    with open(DATA_PATH, 'r', encoding='latin1') as f:
        for linha in f:
            linha = linha.strip()

            if not linha or len(linha) == 1:
                if len(bloco_atual) > 5:
                    blocos.append(bloco_atual)
                bloco_atual = []
                continue

            partes = linha.split()

            if len(partes) >= 2 and '/' in partes[0]:
                try:
                    data_str = partes[0]
                    valor = float(partes[1].replace(',', '.'))
                    bloco_atual.append({'Data': data_str, 'Vibracao': valor})
                except:
                    pass

    if len(bloco_atual) > 5:
        blocos.append(bloco_atual)

    print(f"✅ {len(blocos)} máquinas carregadas")

    blocos_processados = []

    for b in blocos:
        df = pd.DataFrame(b)
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna().sort_values('Data')

        df = df.groupby('Data')['Vibracao'].mean().reset_index()
        df = df.set_index('Data').resample('D').interpolate(method='linear').reset_index()

        blocos_processados.append(df['Vibracao'].values)

    return blocos_processados

# FEATURES

def criar_features_globais(blocos, janela):
    X, y = [], []

    for valores in blocos:
        if len(valores) <= janela:
            continue

        for i in range(len(valores) - janela):
            seq = valores[i:i+janela]
            futuro = valores[i+janela]
            crescimento = futuro - seq[-1]

            if crescimento < -0.05:
                continue

            features = list(seq) + [
                np.mean(seq),
                np.std(seq),
                np.max(seq),
                seq[-1] - seq[0]
            ]

            X.append(features)
            y.append(crescimento)

    return np.array(X), np.array(y)

# PREVISÃO

def projetar_futuro(modelo, valores, janela, passos):
    entrada = valores[-janela:].copy()
    previsoes = []

    for _ in range(passos):
        features = list(entrada) + [
            np.mean(entrada),
            np.std(entrada),
            np.max(entrada),
            entrada[-1] - entrada[0]
        ]

        delta = modelo.predict(np.array(features).reshape(1, -1))[0]

        delta = max(delta, 0)  # evita comportamento irreal

        novo = entrada[-1] + delta
        previsoes.append(novo)

        entrada = np.append(entrada[1:], novo)

    return np.array(previsoes)

# MAIN

def main():
    blocos = carregar_e_limpar_dados()

    X, y = criar_features_globais(blocos, JANELA)

    print(f"🧠 Treinando modelo com {len(X)} exemplos...")
    
    modelo = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )

    modelo.fit(X, y)

    blocos_ordenados = sorted(blocos, key=len, reverse=True)
    top_maquinas = blocos_ordenados[:10]

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n🔎 Analisando máquinas...\n")

    for i, maquina in enumerate(top_maquinas):

        previsoes = projetar_futuro(modelo, maquina, JANELA, PASSOS_FUTURO)

        previsoes_suave = suavizar_curva(previsoes)

        limite = np.mean(maquina) + 2.5 * np.std(maquina)

        falha_idx = detectar_falha(previsoes_suave, limite)

        # ===================== PLOT =====================
        plt.figure(figsize=(12,6))

        plt.plot(maquina, label="Histórico", linewidth=2)

        eixo_futuro = range(len(maquina), len(maquina) + PASSOS_FUTURO)

        plt.plot(eixo_futuro, previsoes, linestyle='dotted', label="Previsão Bruta")

        plt.plot(eixo_futuro, previsoes_suave, linewidth=3, label="Tendência Real")

        plt.axhline(y=limite, color='red', linestyle='--', label="Limite Crítico")

        if falha_idx is not None:
            plt.scatter(len(maquina)+falha_idx,
                        previsoes_suave[falha_idx],
                        color='red', s=100, label="Falha")

        plt.fill_between(eixo_futuro, previsoes_suave, limite,
                         where=(previsoes_suave > limite),
                         alpha=0.3)

        plt.title(f"Máquina {i+1} - Prognóstico de Falha")
        plt.xlabel("Tempo (meses)")
        plt.ylabel("Vibração (RMS)")
        plt.legend()
        plt.grid(alpha=0.3)

        path = os.path.join(PLOTS_DIR, f"maq_{i+1}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        # ===================== RESULTADO =====================
        if falha_idx is not None:
            print(f"⚠️ Máquina {i+1}: falha em ~{falha_idx} meses")
        else:
            print(f"✅ Máquina {i+1}: saudável")

    print("\n🎉 FINALIZADO COM SUCESSO!")


if __name__ == "__main__":
    main()