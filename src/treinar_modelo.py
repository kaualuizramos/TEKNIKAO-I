import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def treinar_ia():
    print("Carregando o histórico das máquinas...")
    df = pd.read_csv('dataset_treino.csv')

    # 1. ENGENHARIA DO GABARITO (TARGET)
    # Na indústria, amplitudes muito altas indicam falha iminente.
    # Vamos criar os rótulos que o botão da Teknikao vai mostrar na tela.
    def classificar_severidade(amplitude):
        if amplitude < 1.0:
            return 'Normal'
        elif amplitude >= 1.0 and amplitude < 5.0:
            return 'Alerta (Acompanhar)'
        else:
            return 'Crítico (Manutenção Imediata)'

    # Cria a coluna 'Status' aplicando a nossa regra de negócio
    df['Status_Real'] = df['Amplitude_Maxima_g'].apply(classificar_severidade)

    # 2. PREPARAÇÃO PARA O MACHINE LEARNING
    # X = O que o modelo usa para prever (Frequência e Amplitude)
    # y = O que o modelo tem que adivinhar (Status_Real)
    X = df[['Frequencia_Pico_Hz', 'Amplitude_Maxima_g']]
    y = df['Status_Real']

    # Divide os dados: 80% para a IA estudar, 20% para aplicarmos a prova final
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TREINAMENTO DO ALGORITMO (Random Forest)
    print("Treinando as Árvores de Decisão...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    # 4. AVALIAÇÃO (O Teste Final)
    previsoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    
    print(f"\nAcurácia da Inteligência Artificial: {acuracia * 100:.2f}%\n")
    print("Relatório de Desempenho do Modelo:")
    print(classification_report(y_teste, previsoes))

    # 5. EXPORTAÇÃO (A Mágica da Integração)
    # Salvamos o "cérebro" treinado em um arquivo .pkl para ser usado depois
    joblib.dump(modelo, 'modelo_teknikao.pkl')
    print(">>> Modelo exportado com sucesso como 'modelo_teknikao.pkl' <<<")
    print("Este é o arquivo que fará a ponte com o Delphi!")

if __name__ == "__main__":
    treinar_ia()