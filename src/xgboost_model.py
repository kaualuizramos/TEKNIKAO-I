
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# PATHS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_treino.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# CONFIG

JANELA = 12
PASSOS_FUTURO = 25

# LOAD DATA

def carregar_dados():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    colunas = df.select_dtypes(include=['float64', 'int64']).columns
    if len(colunas) == 0:
        raise ValueError("❌ Nenhuma coluna numérica encontrada")

    coluna = colunas[0]

    print(f"📊 Coluna: {coluna}")
    print(f"📈 Registros: {len(df)}")

    return df, coluna

# NORMALIZAÇÃO

def normalizar(df, coluna):
    scaler = MinMaxScaler()
    valores = df[coluna].values.reshape(-1, 1)

    valores_scaled = scaler.fit_transform(valores)

    return valores_scaled.flatten(), scaler

# FEATURES INTELIGENTES

def criar_features(valores, janela):
    X, y = [], []

    for i in range(len(valores) - janela):
        seq = valores[i:i+janela]

        features = list(seq)
        features += [
            np.mean(seq),
            np.std(seq),
            np.max(seq),
            np.min(seq),
            seq[-1] - seq[0]  # tendência
        ]

        X.append(features)
        y.append(valores[i+janela])

    return np.array(X), np.array(y)

# TREINAMENTO

def treinar(X, y):
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    print(f"\n📉 MSE: {mse:.6f}")
    print(f"📉 RMSE: {rmse:.6f}")

    return model

# PREVISÃO

def prever_futuro(model, valores, janela, passos):
    entrada = valores[-janela:].copy()
    previsoes = []

    for _ in range(passos):
        features = list(entrada)
        features += [
            np.mean(entrada),
            np.std(entrada),
            np.max(entrada),
            np.min(entrada),
            entrada[-1] - entrada[0]
        ]

        pred = model.predict(np.array(features).reshape(1, -1))[0]
        previsoes.append(pred)

        entrada = np.append(entrada[1:], pred)

    return np.array(previsoes)

# SUAVIZAÇÃO

def suavizar(previsoes, janela=3):
    return np.convolve(previsoes, np.ones(janela)/janela, mode='same')

# DETECÇÃO DE FALHA

def detectar_falha(previsoes):
    contador = 0

    for i in range(1, len(previsoes)):
        crescimento = previsoes[i] - previsoes[i-1]

        if crescimento > 0.01:
            contador += 1
        else:
            contador = 0

        if contador >= 4:
            print("🚨 Tendência consistente de crescimento detectada")
            return i

    return None

# PLOT

def plotar(valores, previsoes, previsoes_suave, scaler):
    valores_real = scaler.inverse_transform(valores.reshape(-1,1)).flatten()
    previsoes_real = scaler.inverse_transform(previsoes.reshape(-1,1)).flatten()
    previsoes_suave_real = scaler.inverse_transform(previsoes_suave.reshape(-1,1)).flatten()

    plt.figure(figsize=(12,6))

    plt.plot(valores_real, label="Dados reais")

    inicio = len(valores_real)
    eixo = range(inicio, inicio + len(previsoes_real))

    plt.plot(eixo, previsoes_real, linestyle="dotted", label="Previsão")
    plt.plot(eixo, previsoes_suave_real, linestyle="dashed", label="Suavizado")

    plt.fill_between(eixo, previsoes_suave_real, alpha=0.2)

    plt.legend()
    plt.title("Previsão de Falha - XGBoost")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "xgb_teknikao.png"))

    plt.show()

# SALVAR MODELO

def salvar_modelo(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(os.path.join(MODELS_DIR, "xgb_model.json"))
    print("\n💾 Modelo salvo com sucesso")

# MAIN

def main():
    print("\n🚀 XGBoost TEKNIKAO - EXECUÇÃO\n")

    try:
        df, coluna = carregar_dados()

        valores, scaler = normalizar(df, coluna)

        X, y = criar_features(valores, JANELA)

        model = treinar(X, y)

        previsoes = prever_futuro(model, valores, JANELA, PASSOS_FUTURO)

        previsoes_suave = suavizar(previsoes)

        falha = detectar_falha(previsoes_suave)

        print("\n================ RESULTADO FINAL ================\n")

        previsoes_real = scaler.inverse_transform(previsoes.reshape(-1,1)).flatten()

        if falha is not None:
            print(f"⚠️ Falha prevista em ~{falha} passos")
            print(f"📊 Valor estimado: {previsoes_real[falha]:.2f}")
            print("\n📌 Interpretação:")
            print("Crescimento consistente indica risco de falha iminente.")

        else:
            print("✅ Nenhuma falha prevista")
            print("\n📌 Interpretação:")
            print("Sistema operando dentro do padrão esperado.")

        plotar(valores, previsoes, previsoes_suave, scaler)

        salvar_modelo(model)

        print("\n✅ Execução concluída com sucesso!")

    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {str(e)}")


if __name__ == "__main__":
    main()