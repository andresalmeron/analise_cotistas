import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuração da Página
st.set_page_config(page_title="Portfel Event Study: Value Add", layout="wide")

st.title("📊 Análise Contrafactual: O 'Alpha' da Recomendação")
st.markdown("""
Esta ferramenta mensura o **Volume Líquido de Cotistas** gerado exclusivamente pela recomendação.
Ela projeta a tendência anterior (Cenário Base) e compara com a realidade (Cenário Real).
""")

# --- FUNÇÕES AUXILIARES ---
def fit_trend_model(df_segment):
    """
    Retorna o modelo treinado, o R2 e as previsões para o segmento.
    """
    if len(df_segment) < 2:
        return None, 0, None, None
    
    # X precisa ser ordinal para regressão matemática
    X = df_segment['Data'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df_segment['Cotistas'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    trend_values = model.predict(X)
    r2 = r2_score(y, trend_values)
    slope = model.coef_[0]
    
    return model, slope, r2, trend_values

# --- UPLOAD E CONFIGURAÇÃO ---
st.sidebar.header("1. Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Envie o arquivo Excel (.xlsx)", type=["xlsx"])

st.sidebar.info("Requer colunas: **Data** e **Cotistas**.")

if uploaded_file is not None:
    try:
        # Carregamento e Tratamento
        df = pd.read_excel(uploaded_file)
        
        cols_needed = ['Data', 'Cotistas']
        if not all(col in df.columns for col in cols_needed):
            st.error(f"Faltam colunas. Necessário: {cols_needed}")
            st.stop()
            
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values(by='Data')
        
        # --- PARÂMETROS ---
        st.sidebar.header("2. Definição do Evento")
        min_date, max_date = df['Data'].min().date(), df['Data'].max().date()
        
        event_date = st.sidebar.date_input(
            "Data da Recomendação",
            value=min_date + (max_date - min_date) // 2,
            min_value=min_date, max_value=max_date
        )
        event_date = pd.to_datetime(event_date)
        
        # --- SEGMENTAÇÃO ---
        df_pre = df[df['Data'] < event_date].copy()
        df_post = df[df['Data'] >= event_date].copy()
        
        if len(df_pre) < 5 or len(df_post) < 2:
            st.error("Dados insuficientes para criar uma regressão robusta. Aumente o período pré-evento.")
            st.stop()

        # --- MODELAGEM E CONTRAFACTUAL ---
        # 1. Treinar no passado (Pré)
        model_pre, slope_pre, r2_pre, trend_pre = fit_trend_model(df_pre)
        
        # 2. Treinar no presente (Pós) apenas para comparar inclinação
        model_post, slope_post, r2_post, trend_post = fit_trend_model(df_post)
        
        # 3. Gerar o Contrafactual (Projetar o modelo PRÉ nas datas PÓS)
        X_post = df_post['Data'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        counterfactual_values = model_pre.predict(X_post) # O que teria acontecido se nada mudasse
        
        # --- CÁLCULO DE VALOR GERADO (ALPHA) ---
        last_actual = df_post['Cotistas'].iloc[-1]
        last_projected = counterfactual_values[-1]
        
        # Alpha Absoluto (Cotistas ganhos "extra")
        net_new_users = last_actual - last_projected
        
        # Alpha Relativo (%)
        uplift_percentage = (net_new_users / last_projected) * 100
        
        # Velocidade
        accel_factor = (slope_post / slope_pre - 1) * 100 if slope_pre != 0 else 0

        # --- DASHBOARD ---
        st.divider()
        
        # KPI Principal: O Alpha
        st.subheader("Resultados de Impacto (Value Add)")
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            st.metric(
                label="Cotistas Reais (Hoje)",
                value=f"{int(last_actual):,}".replace(",", ".")
            )
        
        with kpi2:
            st.metric(
                label="Cenário Projetado (Sem Portfel)",
                value=f"{int(last_projected):,}".replace(",", "."),
                help="Onde o ativo estaria se seguisse a tendência antiga."
            )
            
        with kpi3:
            st.metric(
                label="Alpha Gerado (Cotistas Extras)",
                value=f"{int(net_new_users):+,.0f}".replace(",", "."),
                delta=f"{uplift_percentage:.1f}% vs. Tendência",
                help="Diferença líquida entre o Real e o Projetado."
            )

        # --- VISUALIZAÇÃO ROBUSTA ---
        st.markdown("### Análise Visual da Divergência")
        
        fig = go.Figure()
        
        # 1. Histórico Real (Pontos)
        fig.add_trace(go.Scatter(
            x=df['Data'], y=df['Cotistas'],
            mode='markers', name='Dados Observados',
            marker=dict(color='lightgray', size=5, opacity=0.6),
            showlegend=False
        ))
        
        # 2. Tendência Pré (Linha Base)
        fig.add_trace(go.Scatter(
            x=df_pre['Data'], y=trend_pre,
            mode='lines', name='Tendência Histórica',
            line=dict(color='gray', dash='dot', width=2)
        ))
        
        # 3. Contrafactual (Projeção do Passado no Futuro)
        fig.add_trace(go.Scatter(
            x=df_post['Data'], y=counterfactual_values,
            mode='lines', name='Cenário Contrafactual (Sem Rec.)',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        # 4. Realidade Pós (Linha Sólida)
        fig.add_trace(go.Scatter(
            x=df_post['Data'], y=trend_post, # Usamos a tendência linear do pós para limpar ruído visual
            mode='lines', name='Tendência Real Pós-Rec.',
            line=dict(color='#00CC96', width=4)
        ))
        
        # 5. Preenchimento (O Alpha Visual)
        # Criamos um polígono para pintar a área entre o projetado e o real
        fig.add_trace(go.Scatter(
            x=pd.concat([df_post['Data'], df_post['Data'][::-1]]), # Ida e volta no eixo X
            y=np.concatenate([trend_post, counterfactual_values[::-1]]), # Ida no Y real, volta no Y projetado
            fill='toself',
            fillcolor='rgba(0, 204, 150, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Ganho de Cotistas (Alpha)',
            showlegend=True
        ))

        # Marcador do Evento
        fig.add_vline(x=event_date.timestamp() * 1000, line_width=2, line_color="black")
        fig.add_annotation(x=event_date, y=df['Cotistas'].min(), text="Recomendação", showarrow=False, yshift=10)

        fig.update_layout(
            template="plotly_white",
            height=600,
            xaxis_title="Data",
            yaxis_title="Número de Cotistas",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- RELATÓRIO DE ROBUSTEZ MATEMÁTICA ---
        with st.expander("Ver Detalhes Estatísticos e Robustez"):
            st.markdown(f"""
            **Diagnóstico da Regressão:**
            * **Equação da Tendência Pré:** $y = {slope_pre:.2f}x + C$
            * **Qualidade do Ajuste Pré ($R^2$):** {r2_pre:.4f} (Quanto mais próximo de 1.0, mais confiável é a projeção).
            * **Aceleração da Tendência:** De {slope_pre:.2f} para {slope_post:.2f} cotistas/dia.
            
            *Nota: O modelo assume linearidade no curto prazo. Para janelas de tempo muito longas (> 1 ano), modelos exponenciais seriam preferíveis.*
            """)

    except Exception as e:
        st.error(f"Erro no processamento: {e}")

else:
    st.markdown("---")
    st.markdown("### ⬅️ Faça o upload para iniciar a análise contrafactual.")
