
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

# CONFIG

DATA_PATH = "data/dataset_treino.csv"
JANELA = 20
EPOCHS = 10
BATCH_SIZE = 32

# CARREGAR DADOS

def carregar_dados():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("❌ Dataset vazio!")

    colunas = df.select_dtypes(include=['float64', 'int64']).columns

    if len(colunas) == 0:
        raise ValueError("❌ Nenhuma coluna numérica encontrada")

    coluna = colunas[0]

    print(f"📊 Coluna usada: {coluna}")
    print(f"📈 Total de dados: {len(df)}")

    return df, coluna

# NORMALIZAÇÃO

def normalizar(df, coluna):
    scaler = MinMaxScaler()
    dados = scaler.fit_transform(df[[coluna]])
    return dados, scaler

# SEQUÊNCIAS

def criar_sequencias(data, janela):
    X, y = [], []

    for i in range(len(data) - janela):
        X.append(data[i:i+janela])
        y.append(data[i+janela])

    if len(X) == 0:
        raise ValueError("❌ Dados insuficientes")

    return np.array(X), np.array(y)

# MODELO LSTM

def criar_modelo(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model

# TREINAR

def treinar(model, X, y):
    print("\n🧠 Treinando modelo...\n")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    return model

# PREVER FUTURO

def prever_futuro(model, dados, scaler, passos=50):
    entrada = dados[-JANELA:].copy()
    previsoes = []

    for _ in range(passos):
        entrada_reshape = entrada.reshape((1, JANELA, 1))
        pred = model.predict(entrada_reshape, verbose=0)

        previsoes.append(pred[0][0])
        entrada = np.append(entrada[1:], pred, axis=0)

    previsoes = np.array(previsoes).reshape(-1, 1)
    previsoes = scaler.inverse_transform(previsoes)

    return previsoes

# LIMITE INTELIGENTE

def calcular_limite(df, coluna):
    max_val = df[coluna].max()
    media = df[coluna].mean()

    limite = media + (max_val - media) * 0.7

    print(f"\n📊 Média: {media:.2f}")
    print(f"📊 Máximo: {max_val:.2f}")
    print(f"🚨 Limite calculado: {limite:.2f}")

    return limite

# DETECÇÃO DE FALHA (FINAL)

def detectar_falha(previsoes, limite):
    contador_crescimento = 0

    for i in range(1, len(previsoes)):
        atual = previsoes[i][0]
        anterior = previsoes[i-1][0]

        crescimento = atual - anterior

        # critério 1: valor extremo
        if atual > limite:
            print(f"🚨 Falha crítica (valor alto): {atual:.2f}")
            return i

        # critério 2: crescimento consistente
        if crescimento > 5:
            contador_crescimento += 1
        else:
            contador_crescimento = 0

        if contador_crescimento >= 3:
            print("🚨 Tendência de falha detectada (crescimento contínuo)")
            return i

    return None

# PLOT

def plotar(df, coluna, previsoes, limite):
    plt.figure(figsize=(12,6))

    plt.plot(df[coluna].values, label="Dados reais")

    inicio = len(df)
    eixo_futuro = range(inicio, inicio + len(previsoes))

    plt.plot(eixo_futuro, previsoes, linestyle="dashed", label="Previsão LSTM")

    plt.fill_between(eixo_futuro, previsoes.flatten(), alpha=0.2)

    plt.axhline(y=limite, linestyle='--', label="Limite de falha")

    plt.scatter(inicio, previsoes[0], label="Início previsão")

    plt.legend()
    plt.title("Previsão de Falha com LSTM")

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/lstm_teknikao.png")

    plt.show()

# SALVAR MODELO

def salvar_modelo(model):
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")
    print("\n💾 Modelo salvo em models/lstm_model.keras")

# MAIN

def main():
    print("\n🚀 Iniciando LSTM TEKNIKAO...\n")

    try:
        df, coluna = carregar_dados()

        dados, scaler = normalizar(df, coluna)

        X, y = criar_sequencias(dados, JANELA)

        print(f"\n📐 Shape: {X.shape}")

        model = criar_modelo((X.shape[1], 1))

        model = treinar(model, X, y)

        previsoes = prever_futuro(model, dados, scaler)

        print("\n📊 Primeiras previsões:")
        print(previsoes[:5])

        print("\n📊 Últimos valores previstos:")
        print(previsoes[-5:])

        limite = calcular_limite(df, coluna)

        falha = detectar_falha(previsoes, limite)

        print("\n================ RESULTADO FINAL ================\n")

        if falha is not None:
            valor = previsoes[falha][0]

            print(f"⚠️ Falha prevista em aproximadamente {falha} passos")
            print(f"📊 Valor estimado: {valor:.2f}")

            print("\n📌 Interpretação:")
            print("Comportamento crescente anormal detectado → risco de falha.")

        else:
            print("✅ Nenhuma falha prevista")

            print("\n📌 Interpretação:")
            print("A máquina permanece dentro do padrão normal.")

        plotar(df, coluna, previsoes, limite)

        salvar_modelo(model)

        print("\n✅ Execução finalizada!")

    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")



if __name__ == "__main__":
    main()