import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Portfel Event Study: Linear vs Expo", layout="wide")

st.title("📊 Análise de Impacto: Linear vs. Exponencial")

# --- FUNÇÕES AUXILIARES ---
def fit_trend_model(df_segment, model_type='Linear'):
    """
    Ajusta modelo Linear ou Exponencial.
    """
    if len(df_segment) < 2:
        return None, 0, None, None
    
    # X é sempre ordinal (tempo linear)
    X = df_segment['Data'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df_segment['Cotistas'].values
    
    # Tratamento para modelo Exponencial
    if model_type == 'Exponencial':
        # Evitar log(0) ou log(negativo)
        if np.any(y <= 0):
            return None, 0, None, None # Falha segura
        y_train = np.log(y) # Linearizamos via Log
    else:
        y_train = y
    
    model = LinearRegression()
    model.fit(X, y_train)
    
    # Previsão na escala transformada
    pred_raw = model.predict(X)
    
    # Voltar para escala original se for exponencial
    if model_type == 'Exponencial':
        trend_values = np.exp(pred_raw)
        # Recalcular R2 na escala original (cotistas reais) para ser comparável
        r2 = r2_score(y, trend_values)
    else:
        trend_values = pred_raw
        r2 = r2_score(y, trend_values)
        
    slope = model.coef_[0] # Nota: No expo, isso é a taxa de crescimento % aprox.
    
    return model, slope, r2, trend_values

def project_counterfactual(model, df_post, model_type='Linear'):
    X_post = df_post['Data'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    
    pred_raw = model.predict(X_post)
    
    if model_type == 'Exponencial':
        return np.exp(pred_raw)
    else:
        return pred_raw

# --- INTERFACE ---
st.sidebar.header("1. Configurações")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# SELETOR DE MODELO (A novidade)
model_choice = st.sidebar.radio(
    "Tipo de Crescimento Esperado",
    ["Linear", "Exponencial"],
    help="Linear: Crescimento fixo (cotistas/dia). Exponencial: Crescimento composto (%/dia). Use Exponencial para prazos longos."
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    cols_needed = ['Data', 'Cotistas']
    
    if all(col in df.columns for col in cols_needed):
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values('Data')
        
        # Filtros de Data
        min_date, max_date = df['Data'].min().date(), df['Data'].max().date()
        event_date = st.sidebar.date_input("Data da Recomendação", value=min_date + (max_date - min_date)//2)
        event_date = pd.to_datetime(event_date)
        
        df_pre = df[df['Data'] < event_date].copy()
        df_post = df[df['Data'] >= event_date].copy()
        
        if len(df_pre) > 5 and len(df_post) > 2:
            
            # 1. Modelagem
            try:
                # Ajuste PRÉ (Define o Cenário Base)
                model_pre, slope_pre, r2_pre, trend_pre = fit_trend_model(df_pre, model_choice)
                
                # Ajuste PÓS (Apenas para visualizar a tendência atual)
                model_post, slope_post, r2_post, trend_post = fit_trend_model(df_post, model_choice)
                
                if model_pre is None:
                    st.error("Erro nos dados (possíveis valores zero ou negativos para modelo exponencial).")
                    st.stop()

                # 2. Projeção Contrafactual
                counterfactual = project_counterfactual(model_pre, df_post, model_choice)
                
                # 3. Cálculos de Alpha
                last_real = df['Cotistas'].iloc[-1]
                last_proj = counterfactual[-1]
                alpha_abs = last_real - last_proj
                alpha_pct = (alpha_abs / last_proj) * 100
                
                # --- VISUALIZAÇÃO ---
                st.divider()
                st.subheader(f"Análise usando Modelo {model_choice}")
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Cotistas Reais", f"{int(last_real):,}")
                kpi2.metric("Projeção (Cenário Base)", f"{int(last_proj):,}")
                kpi3.metric("Alpha (Impacto Líquido)", f"{int(alpha_abs):+,}", f"{alpha_pct:.1f}%")
                
                fig = go.Figure()
                
                # Dados
                fig.add_trace(go.Scatter(x=df['Data'], y=df['Cotistas'], mode='markers', name='Observado', marker=dict(color='gray', opacity=0.4)))
                
                # Projeções
                fig.add_trace(go.Scatter(x=df_pre['Data'], y=trend_pre, mode='lines', name='Tendência Pré', line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=df_post['Data'], y=counterfactual, mode='lines', name='Contrafactual (Sem Portfel)', line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=df_post['Data'], y=trend_post, mode='lines', name='Tendência Real', line=dict(color='#00CC96')))
                
                # Área de Alpha
                fig.add_trace(go.Scatter(
                    x=pd.concat([df_post['Data'], df_post['Data'][::-1]]),
                    y=np.concatenate([trend_post, counterfactual[::-1]]),
                    fill='toself', fillcolor='rgba(0,204,150,0.2)', line=dict(width=0), name='Alpha Gerado'
                ))
                
                fig.add_vline(x=event_date.timestamp()*1000, line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explicação Educativa para o Time
                with st.expander("ℹ️ Qual modelo devo escolher?"):
                    st.markdown("""
                    * **Escolha Linear se:** O período de análise é curto (semanas/meses) ou o fundo já é muito maduro e estável.
                    * **Escolha Exponencial se:** O período é longo (> 1 ano) ou o fundo está em fase de crescimento acelerado (early stage). 
                    
                    *Nota: Em fundos de crescimento rápido, o modelo Linear tende a superestimar o Alpha, pois projeta um crescimento base muito lento.*
                    """)
                    
            except Exception as e:
                st.error(f"Erro de cálculo: {e}")
        else:
            st.warning("Dados insuficientes para análise.")
    else:
        st.error("Colunas incorretas.")
