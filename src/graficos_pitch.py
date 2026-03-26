import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# CONFIGURAÇÕES DE DIRETÓRIO
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.txt")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# 1. A FUNÇÃO FÍSICA DE DEGRADAÇÃO (O que o CEO pediu)
def modelo_exponencial(x, a, b, c):
    # a * e^(b*x) + c
    return a * np.exp(b * x) + c

# 2. CARREGAMENTO DOS DADOS (A função que estava faltando antes!)
def carregar_dados_limpos():
    blocos = []
    bloco_atual = []
    with open(DATA_PATH, 'r', encoding='latin1') as f:
        for linha in f:
            linha = linha.strip()
            if not linha or len(linha) == 1:
                if len(bloco_atual) > 5: blocos.append(bloco_atual)
                bloco_atual = []
                continue
            partes = linha.split()
            if len(partes) >= 2 and '/' in partes[0]:
                try:
                    bloco_atual.append({'Data': partes[0], 'Vibracao': float(partes[1].replace(',', '.'))})
                except: pass
    if len(bloco_atual) > 5: blocos.append(bloco_atual)
    return blocos

# 3. LÓGICA PRINCIPAL: ISOLAR A DEGRADAÇÃO E EXTRAPOLAR
def main():
    print("Iniciando a Matemática de Extrapolação Exponencial...")
    blocos = carregar_dados_limpos()
    
    # Filtra máquinas que têm um histórico rico (mais de 15 medições reais)
    maquinas_validas = [b for b in blocos if len(b) > 15]
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    LIMITE_CRITICO = 20.0
    graficos_gerados = 0
    
    for i, maquina_raw in enumerate(maquinas_validas):
        df = pd.DataFrame(maquina_raw)
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna().sort_values('Data').groupby('Data')['Vibracao'].mean().reset_index()
        
        # Interpolação global do ciclo de vida da máquina
        df = df.set_index('Data').resample('D').interpolate(method='linear').reset_index()
        
        # O SEGREDO DO AJUSTE: Isolar o último ciclo de degradação (run-to-failure)
        valores_completos = df['Vibracao'].values
        inicio_ciclo = 0
        for idx in range(len(valores_completos) - 1, 0, -1):
            if valores_completos[idx-1] - valores_completos[idx] > 2.0:
                inicio_ciclo = idx
                break
        
        # Selecionamos APENAS a curva ascendente após a última manutenção
        valores_degradacao = valores_completos[inicio_ciclo:]
        dias = np.arange(len(valores_degradacao))
        
        # Segurança: se após a filtragem não sobrar um período de degradação claro, avançamos
        if len(valores_degradacao) < 30 or (valores_degradacao[-1] - valores_degradacao[0]) < 0.5:
            continue
            
        try:
            # Curve Fitting - bounds forçam a curva a ser estritamente crescente
            popt, _ = curve_fit(modelo_exponencial, dias, valores_degradacao, 
                                p0=(0.1, 0.01, valores_degradacao[0]), 
                                bounds=([0, 0, -np.inf], [np.inf, 1, np.inf]), 
                                maxfev=10000)
            a, b, c = popt
            
            dias_futuros = np.arange(0, len(dias) + 150)
            curva_projetada = modelo_exponencial(dias_futuros, a, b, c)
            
            ponto_de_quebra = np.argmax(curva_projetada >= LIMITE_CRITICO)
            if ponto_de_quebra > 0:
                dias_futuros = dias_futuros[:ponto_de_quebra + 15]
                curva_projetada = curva_projetada[:ponto_de_quebra + 15]
            
            # PLOTAGEM
            plt.figure(figsize=(12, 6))
            plt.plot(dias, valores_degradacao, label="Histórico de Degradação Atual (RMS)", color='blue', linewidth=3)
            plt.plot(dias_futuros[len(dias)-1:], curva_projetada[len(dias)-1:], label="Previsão Exponencial (Matemática Pura)", color='orange', linestyle='--', linewidth=3)
            plt.axhline(y=LIMITE_CRITICO, color='red', linestyle='-', linewidth=2, label=f'Limite Crítico ({LIMITE_CRITICO}g)')
            
            if ponto_de_quebra > 0:
                plt.scatter(ponto_de_quebra, LIMITE_CRITICO, color='red', s=150, zorder=5)
                plt.text(ponto_de_quebra - 20, LIMITE_CRITICO + 1, f"FALHA", color='red', fontweight='bold')

            plt.title(f"Previsão de Colapso - Componente {graficos_gerados+1}", fontsize=14)
            plt.xlabel("Dias de Operação Após Última Manutenção")
            plt.ylabel("Nível de Vibração (RMS)")
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.4)
            
            caminho = os.path.join(PLOTS_DIR, f"grafico_pitch_maquina_limpa_{graficos_gerados+1}.png")
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            plt.close()
            graficos_gerados += 1
            
            if graficos_gerados >= 5: # Limitamos a 5 gráficos espetaculares para o Pitch
                break
                
        except Exception as e:
            pass

    print(f"✅ {graficos_gerados} Gráficos de Pitch gerados com sucesso na pasta {PLOTS_DIR}!")

if __name__ == "__main__":
    main()