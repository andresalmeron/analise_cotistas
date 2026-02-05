import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Portfel Event Study: Robustness", layout="wide")

st.title("📊 Análise de Evento: Impacto e Confiabilidade Estatística")

# --- FUNÇÕES AUXILIARES ---
def fit_trend_model(df_segment, model_type='Linear'):
    """
    Ajusta modelo Linear ou Exponencial e retorna métricas detalhadas.
    """
    if len(df_segment) < 2:
        return None, 0, 0, None, 0
    
    # X é sempre ordinal (tempo linear)
    X = df_segment['Data'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df_segment['Cotistas'].values
    
    # Tratamento para modelo Exponencial
    if model_type == 'Exponencial':
        # Proteção contra log(<=0)
        if np.any(y <= 0):
            return None, 0, 0, None, 0
        y_train = np.log(y) # Linearizamos via Log
    else:
        y_train = y
    
    model = LinearRegression()
    model.fit(X, y_train)
    
    # Previsão na escala transformada
    pred_raw = model.predict(X)
    slope_raw = model.coef_[0]
    
    # Voltar para escala original e calcular métricas interpretáveis
    if model_type == 'Exponencial':
        trend_values = np.exp(pred_raw)
        # R2 calculado sobre os valores REAIS (não sobre os logs) para ser honesto
        r2 = r2_score(y, trend_values)
        # Converter slope logarítmico para taxa de crescimento diária %
        # Fórmula: (e^slope - 1) * 100
        slope_interpretable = (np.exp(slope_raw) - 1) * 100 
    else:
        trend_values = pred_raw
        r2 = r2_score(y, trend_values)
        slope_interpretable = slope_raw # Cotistas/dia
        
    return model, slope_interpretable, r2, trend_values, slope_raw

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

# SELETOR DE MODELO
model_choice = st.sidebar.radio(
    "Modelo de Crescimento Base",
    ["Linear", "Exponencial"],
    help="Define como o ativo se comportaria SEM a recomendação."
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
            
            # --- MODELAGEM ---
            try:
                # Ajuste PRÉ (Define o Cenário Base / Baseline)
                model_pre, slope_pre, r2_pre, trend_pre, raw_slope_pre = fit_trend_model(df_pre, model_choice)
                
                # Ajuste PÓS (Para ver a nova velocidade)
                model_post, slope_post, r2_post, trend_post, raw_slope_post = fit_trend_model(df_post, model_choice)
                
                if model_pre is None:
                    st.error("Erro nos dados (valores nulos ou negativos impedem cálculo exponencial).")
                    st.stop()

                # Projeção Contrafactual (O que aconteceria se o padrão pré continuasse)
                counterfactual = project_counterfactual(model_pre, df_post, model_choice)
                
                # Cálculos de Alpha
                last_real = df['Cotistas'].iloc[-1]
                last_proj = counterfactual[-1]
                alpha_abs = last_real - last_proj
                alpha_pct = (alpha_abs / last_proj) * 100
                
                # --- VISUALIZAÇÃO GRÁFICA ---
                st.subheader(f"Divergência de Tendência ({model_choice})")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Data'], y=df['Cotistas'], mode='markers', name='Observado', marker=dict(color='gray', opacity=0.3, size=5)))
                fig.add_trace(go.Scatter(x=df_pre['Data'], y=trend_pre, mode='lines', name='Tendência Histórica', line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=df_post['Data'], y=counterfactual, mode='lines', name='Contrafactual (Sem Rec.)', line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=df_post['Data'], y=trend_post, mode='lines', name='Tendência Real Pós-Rec.', line=dict(color='#00CC96', width=3)))
                
                # Área de Alpha
                fig.add_trace(go.Scatter(
                    x=pd.concat([df_post['Data'], df_post['Data'][::-1]]),
                    y=np.concatenate([trend_post, counterfactual[::-1]]),
                    fill='toself', fillcolor='rgba(0,204,150,0.2)', line=dict(width=0), name='Alpha Gerado'
                ))
                fig.add_vline(x=event_date.timestamp()*1000, line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)

                # --- DASHBOARD DE CONFIABILIDADE (A Parte Nova) ---
                st.divider()
                st.markdown("### 🕵️ Painel de Auditoria Estatística")
                st.markdown("Aqui validamos se o crescimento é real ou ruído, e a qualidade da nossa projeção.")

                # Organizando em colunas
                col_r2, col_slope, col_verdict = st.columns(3)

                with col_r2:
                    st.markdown("#### 1. Consistência ($R^2$)")
                    st.markdown("Mede o quão 'firme' é a tendência. **Abaixo de 0.50 é fraco**.")
                    
                    st.metric(
                        "Confiabilidade Pré (Baseline)", 
                        f"{r2_pre:.2f}", 
                        help="Se este número for baixo, a 'Projeção Contrafactual' não é confiável, pois o passado era caótico."
                    )
                    st.metric(
                        "Confiabilidade Pós", 
                        f"{r2_post:.2f}",
                        delta=f"{r2_post - r2_pre:.2f}",
                        help="Indica se a nova tendência de alta é consistente ou volátil."
                    )
                    
                    if r2_pre < 0.5:
                        st.warning("⚠️ Atenção: O histórico do ativo é muito volátil. A projeção de 'Alpha' pode estar imprecisa.")

                with col_slope:
                    st.markdown("#### 2. Velocidade (Coef. Angular)")
                    
                    unit_label = "cotistas/dia" if model_choice == "Linear" else "% ao dia"
                    
                    st.metric(
                        "Velocidade Pré", 
                        f"{slope_pre:.3f} {unit_label}"
                    )
                    st.metric(
                        "Velocidade Pós", 
                        f"{slope_post:.3f} {unit_label}",
                        delta=f"{slope_post - slope_pre:.3f}",
                        help="A mudança na velocidade de captação."
                    )

                with col_verdict:
                    st.markdown("#### 3. Veredito Final")
                    st.metric("Alpha Gerado (Total)", f"{int(alpha_abs):+,}", help="Cotistas acima do esperado")
                    st.metric("Uplift (%)", f"{alpha_pct:.1f}%", help="Crescimento percentual sobre o contrafactual")
                    
                    # Lógica de Veredito
                    if alpha_pct > 5 and r2_post > 0.6:
                        st.success("✅ **Sinal Forte:** Aceleração relevante com tendência consistente.")
                    elif alpha_pct > 5 and r2_post <= 0.6:
                        st.warning("⚠️ **Sinal Misto:** Houve crescimento, mas com alta volatilidade (baixa consistência).")
                    elif alpha_pct <= 0:
                        st.error("🔻 **Sem Impacto:** O ativo performou abaixo da tendência histórica.")
                    else:
                        st.info("ℹ️ **Impacto Neutro/Marginal.**")

                # --- ABA EDUCATIVA ---
                st.divider()
                with st.expander("📚 Guia de Bolso: Como interpretar esses indicadores?"):
                    st.markdown("""
                    **1. O Coeficiente de Determinação ($R^2$):**
                    * É a % da variação dos cotistas que é explicada pelo tempo.
                    * **$R^2$ alto (> 0.8):** O crescimento é um "reloginho". Previsível e constante.
                    * **$R^2$ baixo (< 0.4):** O crescimento é caótico. O modelo tem dificuldade em traçar uma reta confiável.
                    * *Insight:* Se o $R^2$ Pré for baixo, não confie cegamente no "Alpha", pois a base de comparação é frágil.

                    **2. Linear vs. Exponencial:**
                    * **Linear:** Assume que o fundo ganha o mesmo nº de cotistas todo dia (Juros Simples). Útil para prazos curtos.
                    * **Exponencial:** Assume que o fundo cresce a uma taxa % composta (Juros Compostos). É o padrão ouro para *startups* e fundos em *ramp-up*.
                    * *Dica:* Se você usar o modelo Linear num período de 2 anos, ele vai "achatada" a curva projetada e inflar artificialmente o seu sucesso. Use o Exponencial para ser conservador e robusto em prazos longos.
                    """)
                    
            except Exception as e:
                st.error(f"Erro ao processar métricas: {e}")
        else:
            st.warning("Dados insuficientes (precisamos de pelo menos 5 pontos pré-evento).")
    else:
        st.error("Colunas incorretas.")
