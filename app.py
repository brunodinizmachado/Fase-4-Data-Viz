import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Configurações Iniciais
st.set_page_config(page_title="HealthAI - Prevenção de Obesidade", layout="wide", page_icon="🩺")

# Carregamento do Modelo
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_obesidade.pkl')
    cols = joblib.load('colunas_modelo.pkl')
    return model, cols

modelo, colunas_treino = load_assets()

# Mapeamentos
niveis_nomes = {
    0: 'Peso Insuficiente', 1: 'Peso Normal', 2: 'Sobrepeso I',
    3: 'Sobrepeso II', 4: 'Obesidade I', 5: 'Obesidade II', 6: 'Obesidade III'
}

# --- INTERFACE ---
st.title("🩺 Hospital Digital: Predição e Análise de Obesidade")
st.markdown("---")

# Criando as abas pedidas no requisito
tab_pred, tab_analise = st.tabs(["🚀 Aplicação Preditiva (Deploy)", "📊 Visão Analítica (Insights Médicos)"])

# --- ABA 1: DEPLOY DO MODELO ---
with tab_pred:
    st.subheader("Nova Avaliação Preventiva")
    st.write("Preencha os dados comportamentais para identificar o risco do paciente.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gênero", ("Feminino", "Masculino"))
        age = st.slider("Idade", 14, 65, 25)
        family = st.selectbox("Histórico Familiar de Sobrepeso?", ("Sim", "Não"))
        caec = st.selectbox("Come entre as refeições?", ("Não", "Às vezes", "Frequentemente", "Sempre"))
        faf = st.slider("Atividade Física (dias/semana)", 0, 3, 1)

    with col2:
        favc = st.selectbox("Consome alimentos calóricos com frequência?", ("Sim", "Não"))
        fcvc = st.slider("Consumo de Vegetais (1: Pouco, 3: Muito)", 1, 3, 2)
        ch2o = st.slider("Consumo de Água (Litros/dia)", 1, 3, 2)
        scc = st.selectbox("Monitora Calorias?", ("Sim", "Não"))
        calc = st.selectbox("Consumo de Álcool", ("Não", "Às vezes", "Frequentemente", "Sempre"))

    if st.button("Executar Diagnóstico Preditivo"):
        # Preparação dos dados (idêntico ao treino)
        input_data = {
            'Gender': 1 if gender == "Masculino" else 0,
            'Age': age,
            'family_history': 1 if family == "Sim" else 0,
            'FAVC': 1 if favc == "Sim" else 0,
            'FCVC': fcvc,
            'NCP': 3, # Valor médio padrão
            'CAEC': {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}[caec],
            'SMOKE': 0,
            'CH2O': ch2o,
            'SCC': 1 if scc == "Sim" else 0,
            'FAF': faf,
            'TUE': 1,
            'CALC': {"Não": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}[calc],
            'transporte_Automobile': 0, 'transporte_Bike': 0, 'transporte_Motorbike': 0,
            'transporte_Public_Transportation': 1, 'transporte_Walking': 0
        }
        
        df_input = pd.DataFrame(input_data, index=[0])[colunas_treino]
        pred = modelo.predict(df_input)[0]
        prob = modelo.predict_proba(df_input).max()
        
        st.metric("Resultado:", niveis_nomes[pred])
        st.info(f"Confiança da Predição: {prob:.2%}")

# --- ABA 2: VISÃO ANALÍTICA ---
with tab_analise:
    st.subheader("Painel de Insights para Equipe Médica")
    st.write("Estudo baseado em 2.111 casos reais.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia do Modelo", "77.51%", "Foco Preventivo")
    c2.metric("Principal Fator", "Genética", "Corr: 0.50")
    c3.metric("Público Crítico", "Mulheres", "Obesidade III")

    st.markdown("---")
    
    st.subheader("💡 Insights Estratégicos")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🧬 Genética vs Comportamento")
        st.write("""
        O histórico familiar é o preditor mais forte. Pacientes com 'family_history' positivo 
        devem entrar em protocolos de monitoramento SCC (Contagem de Calorias) imediatamente.
        """)
        
    with col_b:
        st.markdown("### 🍔 O Paradoxo do CAEC")
        st.write("""
        Dados mostram que o hábito de 'beliscar' (CAEC) reportado como 'Sempre' é menos comum 
        nos níveis de obesidade severa do que o 'Às Vezes', sugerindo subnotificação ou 
        mudança na qualidade calórica das refeições principais.
        """)

    st.warning("Nota Técnica: Este modelo não utiliza Peso e Altura (IMC), focando exclusivamente em variáveis de estilo de vida para suporte à decisão preventiva.")