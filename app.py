import streamlit as st
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# CONFIG

st.set_page_config(page_title="TEKNIKAO Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.txt")

JANELA = 15
PASSOS_FUTURO = 120
MAX_MAQUINAS = 50

# FEATURES

def gerar_features(seq):
    return list(seq) + [
        np.mean(seq),
        np.std(seq),
        np.max(seq),
        seq[-1] - seq[0]
    ]

# SUAVIZAÇÃO

def suavizar_curva(dados, alpha=0.3):
    suavizado = [dados[0]]
    for i in range(1, len(dados)):
        suavizado.append(alpha * dados[i] + (1 - alpha) * suavizado[-1])
    return np.array(suavizado)

# DETECÇÃO

def detectar_falha(previsoes, limite):
    for i, v in enumerate(previsoes):
        if v > limite:
            return i
    return None

# CARREGAR DADOS

@st.cache_data
def carregar_dados():
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
                    valor = float(partes[1].replace(',', '.'))
                    bloco_atual.append(valor)
                except:
                    pass

    if len(bloco_atual) > 5:
        blocos.append(bloco_atual)

    return blocos

# FEATURES GLOBAIS

def criar_features(blocos):
    X, y = [], []

    for valores in blocos:
        if len(valores) <= JANELA:
            continue

        for i in range(len(valores) - JANELA):
            seq = valores[i:i+JANELA]
            futuro = valores[i+JANELA]
            crescimento = futuro - seq[-1]

            if crescimento < -0.05:
                continue

            X.append(gerar_features(seq))
            y.append(crescimento)

    return np.array(X), np.array(y)

# TREINAR

@st.cache_resource
def treinar_modelo(X, y):
    modelo = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    modelo.fit(X, y)
    return modelo

# PREVISÃO CORRIGIDA

def projetar(modelo, valores):
    entrada = valores[-JANELA:].copy()
    previsoes = []

    limite = np.mean(valores) + 2.5 * np.std(valores)

    for _ in range(PASSOS_FUTURO):
        features = gerar_features(entrada)
        features = np.array(features).reshape(1, -1)

        delta = modelo.predict(features)[0]

        # impedir negativos
        delta = max(delta, 0)

        # limitar crescimento absurdo
        delta = np.clip(delta, 0, 2.0)

        # suavizar crescimento
        delta = delta * 0.7

        novo = entrada[-1] + delta

        # limite físico
        novo = min(novo, limite * 1.5)

        previsoes.append(novo)

        entrada = np.append(entrada[1:], novo)

    return np.array(previsoes)

# APP

st.title("🔧 TEKNIKAO - Monitoramento Inteligente")

st.write("Sistema de manutenção preditiva com IA")

# carregar dados
blocos_brutos = carregar_dados()

# filtrar válidos
blocos_validos = [b for b in blocos_brutos if len(b) > JANELA]

# ordenar
blocos_validos = sorted(blocos_validos, key=len, reverse=True)

# limitar
blocos_validos = blocos_validos[:MAX_MAQUINAS]

# features globais
X, y = criar_features(blocos_validos)

# treino
with st.spinner("Treinando modelo..."):
    modelo = treinar_modelo(X, y)

# seleção
indice = st.selectbox(
    "Selecione a máquina",
    range(len(blocos_validos)),
    format_func=lambda x: f"Máquina {x+1} ({len(blocos_validos[x])} pontos)"
)

maquina = blocos_validos[indice]

# previsão
previsoes = projetar(modelo, maquina)
previsoes_suave = suavizar_curva(previsoes)

# limite
limite = np.mean(maquina) + 2.5 * np.std(maquina)

# falha
falha_idx = detectar_falha(previsoes_suave, limite)

# STATUS

st.subheader("📊 Status da Máquina")

if falha_idx is not None:
    if falha_idx <= 3:
        st.error(f"🔴 CRÍTICO - Falha em ~{falha_idx} meses")
    elif falha_idx <= 6:
        st.warning(f"🟡 ALERTA - Falha em ~{falha_idx} meses")
    else:
        st.info(f"🟢 Estável (falha distante: ~{falha_idx} meses)")
else:
    st.success("🟢 Máquina saudável")

# GRÁFICO

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(maquina, label="Histórico", linewidth=2)

eixo_futuro = range(len(maquina), len(maquina)+PASSOS_FUTURO)

ax.plot(eixo_futuro, previsoes_suave,
        linestyle='--', linewidth=2, label="Previsão")

ax.axhline(y=limite, linestyle=':', linewidth=2, label="Limite Crítico")

if falha_idx is not None:
    ax.scatter(len(maquina)+falha_idx,
               previsoes_suave[falha_idx],
               s=100)

ax.legend()
ax.set_title("Prognóstico de Falha")
ax.set_xlabel("Tempo (meses)")
ax.set_ylabel("Vibração")

st.pyplot(fig)