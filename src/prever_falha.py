import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta

# 1. A FUNÇÃO MATEMÁTICA INOVADORA (Exponencial em vez de Polinomial)
# Modela o desgaste real: degradação lenta no início, aceleração rápida no final.
def modelo_exponencial(x, a, b, c):
    return a * np.exp(b * x) + c

def projetar_falha_maquina():
    print("Iniciando o Algoritmo de Prognóstico AAV...")
    
    # ---------------------------------------------------------
    # DADOS SIMULADOS (Substitua isso pelo CSV que a Teknikao vai enviar)
    # ---------------------------------------------------------
    datas = pd.date_range(start='2025-01-01', periods=10, freq='15D')
    # Simulando a vibração subindo aos poucos
    vibracao_rms = [0.5, 0.52, 0.55, 0.6, 0.68, 0.8, 0.95, 1.2, 1.6, 2.2]
    
    df = pd.DataFrame({'Data': datas, 'RMS': vibracao_rms})
    
    # O limite que o dono da máquina escolhe como perigoso
    LIMITE_CRITICO = 4.5  

    # 2. PREPARAÇÃO DO EIXO X (Transformando Datas em Números)
    # A matemática precisa de dias (0, 15, 30...), não de formatos de data civil
    data_inicial = df['Data'].iloc[0]
    df['Dias_Operacao'] = (df['Data'] - data_inicial).dt.days
    
    x_conhecido = df['Dias_Operacao'].values
    y_conhecido = df['RMS'].values

    # 3. O "CURVE FITTING" (Ajuste da Curva)
    # O Scipy descobre sozinho os melhores valores de a, b e c para a nossa curva
    try:
        parametros_otimizados, _ = curve_fit(modelo_exponencial, x_conhecido, y_conhecido, p0=(0.1, 0.01, 0.5))
        a, b, c = parametros_otimizados
    except Exception as e:
        print("Erro ao ajustar a curva. Precisamos de mais dados.")
        return

    # 4. A EXTRAPOLAÇÃO PARA O FUTURO (Previsão)
    # Vamos criar "dias futuros" para ver onde a linha vai bater
    dias_futuros = np.arange(0, x_conhecido[-1] + 100, 1) # Projeta 100 dias pra frente
    previsao_rms = modelo_exponencial(dias_futuros, a, b, c)

    # 5. ENCONTRANDO A DATA DA FALHA
    # Acha o exato dia em que a curva cruza o Limite Crítico
    dia_da_falha = dias_futuros[np.argmax(previsao_rms >= LIMITE_CRITICO)]
    data_da_falha = data_inicial + timedelta(days=int(dia_da_falha))

    # 6. PLOTANDO O GRÁFICO (A Entrega Visual)
    plt.figure(figsize=(10, 6))
    
    # Plota os pontos reais (Passado)
    plt.scatter(df['Data'], y_conhecido, color='blue', label='Histórico Conhecido', zorder=5)
    
    # Plota a curva do futuro
    datas_projetadas = [data_inicial + timedelta(days=int(d)) for d in dias_futuros]
    plt.plot(datas_projetadas, previsao_rms, color='orange', linestyle='--', label='Projeção Exponencial (IA)')
    
    # Desenha a linha do Limite de Perigo
    plt.axhline(y=LIMITE_CRITICO, color='red', linestyle='-', label=f'Limite Crítico ({LIMITE_CRITICO}g)')
    
    # Marca o ponto exato da quebra
    plt.scatter(data_da_falha, LIMITE_CRITICO, color='red', s=100, marker='X', zorder=6)
    plt.annotate(f'Falha Estimada:\n{data_da_falha.strftime("%d/%m/%Y")}', 
                 xy=(data_da_falha, LIMITE_CRITICO), 
                 xytext=(-80, 20), textcoords='offset points', color='red', weight='bold')

    plt.title('Prognóstico de Falha Mecânica - Modelo AAV')
    plt.xlabel('Data')
    plt.ylabel('Vibração Global (RMS)')
    plt.legend()
    plt.grid(True)
    plt.savefig('grafico_previsao.png')
    
    print(f"O algoritmo calculou que a máquina atingirá o nível crítico em: {data_da_falha.strftime('%d/%m/%Y')}")
    print("Gráfico 'grafico_previsao.png' gerado com sucesso!")

if __name__ == "__main__":
    projetar_falha_maquina()