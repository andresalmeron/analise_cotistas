import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Portfel Research: Event Study", layout="wide")

st.title("📊 Análise de Evento: Impacto e Confiabilidade")

# --- 1. FUNÇÕES AUXILIARES ---

def clean_outliers_anbima(df):
    """
    Remove ruídos abruptos (ex: quedas para 1 cotista) típicos de bases CVM/Anbima.
    Usa Mediana Móvel para não ser afetado pelos próprios outliers.
    """
    df_clean = df.copy()
    
    # 1. Calcula a Mediana Móvel de 5 dias (suaviza a tendência local)
    # Usamos mediana pois ela ignora o '1' no meio de vários '180'
    median_rolling = df_clean['Cotistas'].rolling(window=5, center=True, min_periods=1).median()
    
    # 2. Identifica desvios bruscos (>50% de queda ou >50% de alta súbita sobre a mediana)
    # Isso pega o caso 180 -> 1 (que é < 50% de 180)
    is_outlier = (df_clean['Cotistas'] < median_rolling * 0.5) | \
                 (df_clean['Cotistas'] > median_rolling * 1.5)
    
    # 3. Substitui por NaN e Interpola Linearmente
    if is_outlier.sum() > 0:
        df_clean.loc[is_outlier, 'Cotistas'] = np.nan
        df_clean['Cotistas'] = df_clean['Cotistas'].interpolate(method='linear', limit_direction='both')
        
    return df_clean, is_outlier.sum()

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

# --- 2. CONFIGURAÇÃO E UPLOAD ---
st.sidebar.header("1. Upload e Limpeza")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# Opção de Limpeza (Já vem marcada como True por segurança)
do_clean = st.sidebar.checkbox("Ativar Higienização de Dados (Anbima)", value=True, help="Corrige falhas como quedas para 1 cotista.")

# Opção de Modelo
st.sidebar.header("2. Modelagem")
model_choice = st.sidebar.radio(
    "Modelo de Crescimento Base",
    ["Linear", "Exponencial"],
    help="Linear: Juros Simples. Exponencial: Juros Compostos (longo prazo)."
)

if uploaded_file:
    # Leitura Inicial
    df_raw = pd.read_excel(uploaded_file)
    cols_needed = ['Data', 'Cotistas']
    
    if all(col in df_raw.columns for col in cols_needed):
        df_raw['Data'] = pd.to_datetime(df_raw['Data'])
        df_raw = df_raw.sort_values('Data')
        
        # --- APLICAÇÃO DA LIMPEZA ---
        if do_clean:
            df, num_errors = clean_outliers_anbima(df_raw)
            if num_errors > 0:
                st.sidebar.success(f"✅ {num_errors} outliers corrigidos.")
                
                # Visualização Opcional do "Antes vs Depois"
                with st.sidebar.expander("Ver Correções"):
                    st.write("Dados corrigidos automaticamente.")
                    chart_comp = go.Figure()
                    chart_comp.add_trace(go.Scatter(x=df_raw['Data'], y=df_raw['Cotistas'], name='Original (Erro)', line=dict(color='red', width=1)))
                    chart_comp.add_trace(go.Scatter(x=df['Data'], y=df['Cotistas'], name='Limpo', line=dict(color='green', width=2)))
                    chart_comp.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(chart_comp, use_container_width=True)
        else:
            df = df_raw.copy()
            num_errors = 0

        # --- SELEÇÃO DE DATAS ---
        min_date, max_date = df['Data'].min().date(), df['Data'].max().date()
        event_date = st.sidebar.date_input("Data da Recomendação", value=min_date + (max_date - min_date)//2)
        event_date = pd.to_datetime(event_date)
        
        df_pre = df[df['Data'] < event_date].copy()
        df_post = df[df['Data'] >= event_date].copy()
        
        if len(df_pre) > 5 and len(df_post) > 2:
            
            # --- MODELAGEM E CÁLCULOS ---
            try:
                # Ajuste PRÉ (Baseline)
                model_pre, slope_pre, r2_pre, trend_pre, raw_slope_pre = fit_trend_model(df_pre, model_choice)
                
                # Ajuste PÓS (Tendência Atual)
                model_post, slope_post, r2_post, trend_post, raw_slope_post = fit_trend_model(df_post, model_choice)
                
                if model_pre is None:
                    st.error("Erro nos dados: valores nulos ou negativos impedem cálculo exponencial.")
                    st.stop()

                # Contrafactual (O "E Se...")
                counterfactual = project_counterfactual(model_pre, df_post, model_choice)
                
                # Alpha (Impacto)
                last_real = df['Cotistas'].iloc[-1]
                last_proj = counterfactual[-1]
                alpha_abs = last_real - last_proj
                alpha_pct = (alpha_abs / last_proj) * 100
                
                # --- VISUALIZAÇÃO PRINCIPAL ---
                st.subheader(f"Divergência de Tendência ({model_choice})")
                
                fig = go.Figure()
                # Pontos (Dados Reais Limpos)
                fig.add_trace(go.Scatter(x=df['Data'], y=df['Cotistas'], mode='markers', name='Observado', marker=dict(color='gray', opacity=0.3, size=5)))
                # Linhas de Tendência
                fig.add_trace(go.Scatter(x=df_pre['Data'], y=trend_pre, mode='lines', name='Tendência Histórica', line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=df_post['Data'], y=counterfactual, mode='lines', name='Contrafactual (Sem Rec.)', line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=df_post['Data'], y=trend_post, mode='lines', name='Tendência Real Pós-Rec.', line=dict(color='#00CC96', width=3)))
                
                # Área de Alpha (Verde)
                fig.add_trace(go.Scatter(
                    x=pd.concat([df_post['Data'], df_post['Data'][::-1]]),
                    y=np.concatenate([trend_post, counterfactual[::-1]]),
                    fill='toself', fillcolor='rgba(0,204,150,0.2)', line=dict(width=0), name='Alpha Gerado'
                ))
                # Linha Vertical do Evento
                fig.add_vline(x=event_date.timestamp()*1000, line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)

                # --- DASHBOARD DE AUDITORIA ---
                st.divider()
                st.markdown("### 🕵️ Painel de Auditoria Estatística")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("#### 1. Consistência ($R^2$)")
                    st.metric("Confiabilidade Pré", f"{r2_pre:.2f}".replace('.',','), help="Se baixo (<0.4), o passado era caótico.")
                    st.metric("Confiabilidade Pós", f"{r2_post:.2f}".replace('.',','), delta=f"{r2_post - r2_pre:.2f}")
                    if r2_pre < 0.4:
                        st.warning("⚠️ Histórico volátil. Projeção baseada em ruído.")

                with c2:
                    st.markdown("#### 2. Velocidade")
                    lbl = "cotistas/dia" if model_choice == "Linear" else "% ao dia"
                    st.metric("Velocidade Pré", f"{slope_pre:.2f}".replace('.',',') + f" {lbl}")
                    st.metric("Velocidade Pós", f"{slope_post:.2f}".replace('.',',') + f" {lbl}", delta=f"{slope_post - slope_pre:.3f}")

                with c3:
                    st.markdown("#### 3. Veredito Final")
                    st.metric("Alpha (Cotistas)", f"{int(alpha_abs):+,}".replace(',',' '))
                    st.metric("Uplift (%)", f"{alpha_pct:.1f}%")
                    
                    if alpha_pct > 5 and r2_post > 0.6:
                        st.success("✅ **Sinal Forte:** Aceleração Real.")
                    elif alpha_pct > 0:
                        st.warning("⚠️ **Sinal Misto:** Crescimento com ruído.")
                    else:
                        st.error("🔻 **Sem Impacto.**")

                # --- ABA EDUCATIVA ---
                st.divider()
                with st.expander("📚 Notas Técnicas"):
                    st.markdown("""
                    * **Limpeza de Dados:** Ativada. Removemos saltos irreais (ex: 180->1) usando mediana móvel.
                    * **$R^2$ (R-Quadrado):** Mede o quão 'firme' é a linha de tendência. Se $R^2$ Pré for negativo ou muito baixo, significa que o ativo não tinha direção definida antes da recomendação.
                    * **Modelo Exponencial:** Recomendado para análises de longo prazo (>1 ano) para capturar o efeito de juros compostos no crescimento da base.
                    """)
                    
            except Exception as e:
                st.error(f"Erro de processamento: {e}")
        else:
            st.warning("Dados insuficientes para análise estatística (poucos dias pré/pós evento).")
    else:
        st.error("Erro: A planilha precisa ter as colunas 'Data' e 'Cotistas'.")
