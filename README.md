# 🔧 TEKNIKAO - Sistema de Manutenção Preditiva com IA

## 📌 Visão Geral

Este projeto implementa um sistema de **manutenção preditiva** baseado em análise de vibração de máquinas industriais.

Utilizando técnicas de **Machine Learning (XGBoost)**, o sistema é capaz de:

* 📊 Analisar dados históricos de vibração
* 🔮 Prever o comportamento futuro da máquina
* 🚨 Detectar falhas antes que ocorram
* ⏳ Estimar o tempo restante até a falha (RUL - Remaining Useful Life)

---

## 🧠 Como Funciona

O sistema segue 4 etapas principais:

### 1. 📥 Coleta e Processamento de Dados

* Leitura de dados de vibração ao longo do tempo
* Organização por ciclos de vida de máquinas
* Limpeza e preparação dos dados

---

### 2. 🧠 Engenharia de Features

Para cada janela de tempo (15 pontos), são extraídas:

* Valores históricos da vibração
* Valor de vibração global
* Desvio padrão
* Valor máximo
* Tendência (crescimento da vibração)

---

### 3. 🤖 Modelagem com XGBoost

O modelo aprende a prever:

> 📈 **A taxa de crescimento da vibração (Δv)**

Em vez de prever diretamente o valor, o sistema modela a evolução do desgaste.

---

### 4. 🔮 Projeção de Falha

A previsão é feita de forma iterativa:

* O modelo prevê o crescimento
* O valor é atualizado
* O processo se repete para simular o futuro

---

## 📊 Interpretação dos Gráficos

O dashboard apresenta:

* 🔵 **Histórico (linha azul)**
  Representa o comportamento real da máquina

* 🟠 **Previsão (linha tracejada)**
  Projeção futura feita pela IA

* 🔴 **Limite crítico**
  Calculado como:
  `média + 2.5 * desvio padrão`

* 🔴 **Ponto de falha**
  Indica quando a máquina ultrapassa o limite seguro

---

## 🚨 Classificação de Risco

O sistema classifica automaticamente o estado da máquina:

| Status      | Interpretação           |
| ----------- | ----------------------- |
| 🟢 Saudável | Operação normal         |
| 🟡 Alerta   | Tendência de degradação |
| 🔴 Crítico  | Falha iminente          |

---

## 📈 Lógica Matemática

O modelo utiliza:

* Derivada discreta:
  Δv = v(t+1) - v(t)

* Regressão com XGBoost:
  Δv = f(features)

* Projeção futura:
  v(t+1) = v(t) + Δv

* Detecção de falha:
  v > μ + 2.5σ

---

## ⚙️ Tecnologias Utilizadas

* Python
* XGBoost
* NumPy
* Matplotlib
* Streamlit

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

---

### 2. Rodar o dashboard

```bash
streamlit run app.py
```

---

## 📁 Estrutura do Projeto

```
📦 TEKNIKAO-I
 ┣ 📂 data/               # Ficheiros de dados brutos e processados (Dataset.txt)
 ┣ 📂 models/             # Cérebro da IA exportado (.pkl, .json, .keras)
 ┣ 📂 plots/              # Saída dos algoritmos (Gráficos RUL, Curvas de Tendência)
 ┣ 📂 reports/            # Relatórios analíticos exportados
 ┣ 📂 src/                # Código-fonte principal da solução
 ┃ ┣ 📜 DatasetCleaner.py         # ETL e Interpolação de dados
 ┃ ┣ 📜 graficos_pitch.py         # Ajuste Físico-Matemático Exponencial
 ┃ ┣ 📜 complete_pipeline.py      # Execução orquestrada do sistema
 ┃ ┗ 📜 xgboost_model_v4.py       # Modelo Preditivo ML
 ┣ 📜 requirements.txt    # Dependências do projeto
 ┗ 📜 README.md           # Documentação do projeto
```

---

## 💡 Diferenciais do Projeto

* ✔️ Modelagem baseada em crescimento (não apenas valor)
* ✔️ Previsão temporal realista
* ✔️ Detecção antecipada de falhas
* ✔️ Dashboard interativo
* ✔️ Aplicação prática industrial

---

## 🎯 Aplicações Reais

* Indústria 4.0
* Monitoramento de máquinas
* Manutenção preditiva
* Redução de downtime
* Otimização de custos

---

## 🧠 Conclusão

Este projeto demonstra como técnicas de **Machine Learning + Estatística** podem ser aplicadas para transformar dados em decisões estratégicas.

> 🔥 De manutenção corretiva → para manutenção preditiva

---

## 👨‍💻 Autor

Projeto desenvolvido por:

**Kauã Luiz Ramos**
**Lucas Tavares Vieira**
**Rafael Souza Katahira**

---
