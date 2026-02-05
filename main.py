import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

# Configuración de la página
st.set_page_config(page_title="BTC Power Law", page_icon="🪙", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        h1 {text-align: center;}
        .stHeading {padding-bottom: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# Título
st.title("🪙 Bitcoin Power Law 🪙")

# Función para cargar datos
@st.cache_data(ttl="1d")
def load_data():
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=csv"
    df_blockchain = pd.read_csv(url, header=None, names=['Fecha', 'Precio'])
    df_blockchain['Fecha'] = pd.to_datetime(df_blockchain['Fecha'])
    df_blockchain = df_blockchain[df_blockchain['Precio'] > 0].copy()
    
    # Transformaciones
    genesis_date = pd.to_datetime('2009-01-03')
    df_blockchain['DiasDesdeGenesis'] = (df_blockchain['Fecha'] - genesis_date).dt.days + 1
    df_blockchain['log_DiasDesdeGenesis'] = np.log10(df_blockchain['DiasDesdeGenesis'])
    df_blockchain['log_Precio'] = np.log10(df_blockchain['Precio'])
    
    return df_blockchain

# Cargar datos
try:
    with st.spinner('Cargando datos de blockchain.com...'):
        df = load_data()
    
    # Cuantiles y colores fijos
    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    
    # Variables para el modelo en escala logarítmica
    x_log = df['log_DiasDesdeGenesis']
    y_log = df['log_Precio']
    X_log = sm.add_constant(x_log)
    
    # Variables para el modelo en escala nominal
    x_nominal = df['DiasDesdeGenesis']
    y_nominal = df['Precio']
    
    # Ajustar modelos
    models_log = []
    models_nominal = []
    
    for q in quantiles:
        # Modelo logarítmico
        model_log = sm.QuantReg(y_log, X_log)
        res_log = model_log.fit(q=q)
        models_log.append(res_log)
        
        # Modelo nominal (usando las mismas variables log para consistencia)
        model_nominal = sm.QuantReg(y_log, X_log)
        res_nominal = model_nominal.fit(q=q)
        models_nominal.append(res_nominal)
    
    # Crear tabs
    tab1, tab2 = st.tabs(["📈 Logarithmic", "💵 Nominal (USD)"])
    
    with tab1:
        # Gráfico logarítmico
        fig_log = go.Figure()
        
        # Scatter plot de datos
        fig_log.add_trace(go.Scatter(
            x=x_log,
            y=y_log,
            mode='markers',
            name='Data',
            marker=dict(color='lightgray', size=3, opacity=0.5),
            hovertemplate='Log(Days): %{x:.3f}<br>Log(Price): %{y:.3f}<extra></extra>'
        ))
        
        # Líneas de regresión cuantílica
        for i, (q, color) in enumerate(zip(quantiles, colors)):
            res = models_log[i]
            y_pred = res.predict(X_log)
            slope = res.params.iloc[1]
            intercept = res.params.iloc[0]
            
            fig_log.add_trace(go.Scatter(
                x=x_log,
                y=y_pred,
                mode='lines',
                name=f'Q{q*100:.0f}',
                line=dict(color=color, width=2),
                hovertemplate=f'Q{q*100:.0f}<br>Log(Days): %{{x:.3f}}<br>Log(Price): %{{y:.3f}}<extra></extra>'
            ))
        
        fig_log.update_layout(
            # title='Regresiones Cuantílicas - Escala Log-Log',
            xaxis_title='Log₁₀(Days since Genesis Bolck)',
            yaxis_title='Log₁₀(Price in USD)',
            hovermode='closest',
            height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
             margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig_log, use_container_width=True)
    
    with tab2:
        # Gráfico nominal
        fig_nominal = go.Figure()

        # Scatter plot de datos en escala nominal
        fig_nominal.add_trace(go.Scatter(
            x=df['Fecha'],
            y=df['Precio'],
            mode='markers',
            name='Data',
            marker=dict(color='lightgray', size=3, opacity=0.5),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # Líneas de regresión cuantílica convertidas a escala nominal
        for i, (q, color) in enumerate(zip(quantiles, colors)):
            res = models_nominal[i]
            y_pred_log = res.predict(X_log)
            y_pred_nominal = 10 ** y_pred_log  # Convertir de log a nominal
            slope = res.params.iloc[1]
            intercept = res.params.iloc[0]
            
            fig_nominal.add_trace(go.Scatter(
                x=df['Fecha'],
                y=y_pred_nominal,
                mode='lines',
                name=f'Q{q*100:.0f}',
                line=dict(color=color, width=2),
                hovertemplate=f'Q{q*100:.0f}<br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.2f}}<extra></extra>'
            ))
        
        fig_nominal.update_layout(
            # title='Regresiones Cuantílicas - Escala Nominal',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='closest',
            height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis_type='log',  # Eje Y logarítmico para mejor visualización
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig_nominal, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")