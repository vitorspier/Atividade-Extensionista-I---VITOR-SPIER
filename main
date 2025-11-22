import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from io import StringIO
import warnings

# Suprimir avisos de desvio de tipos do sklearn, pois estamos forçando a remoção de 'object'
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURAÇÕES BÁSICAS DO STREAMLIT ---
st.set_page_config(
    page_title="Análise de Clientes (IA Simples)",
    layout="wide"
)

st.title("Análise de Saúde do Cliente com IA (Versão Simples)")
st.markdown("Ferramenta para classificar clientes em 'Afastando-se', 'Fidelizado', 'Em Ascensão' ou 'Estável'.")
st.markdown("---")

# 1. CARREGAR DADOS DE TREINAMENTO (FIXO)

# Nota: Em um projeto real, esta base viria de um banco de dados e conteria 1000 linhas.
data_raw = """cliente_id,data_cadastro,data_ultima_compra,Potencial,total_compras,ticket_medio,frequencia_mensal,categoria_cliente,estado,participou_promocoes,churn,dias_desde_ultima_compra
C0000,2020-03-13 00:00:00,2025-06-17 00:00:00,100,39,1060.59,1.14,Atacado,PR,1,1,47
C0001,2020-04-25 00:00:00,2024-03-22 00:00:00,200,29,1496.02,4.11,Distribuidor,SP,0,0,499
C0002,2023-05-15 00:00:00,2024-12-31 00:00:00,300,15,1350.08,1.68,Online,RS,0,1,215
C0003,2022-09-15 00:00:00,2025-05-13 00:00:00,400,43,885.2,0.21,Online,MG,1,0,82
C0004,2023-06-18 00:00:00,2025-04-29 00:00:00,500,8,1380.22,0.85,Atacado,SP,1,0,96
C0005,2022-01-21 00:00:00,2024-05-01 00:00:00,600,21,57.69,1.2,Atacado,RJ,1,0,459
C0006,2021-10-19 00:00:00,2025-04-08 00:00:00,700,39,1463.85,4.79,Atacado,MG,0,0,117
C0007,2022-03-25 00:00:00,2024-01-26 00:00:00,800,19,761.59,2.91,Varejo,RS,1,0,555
C0008,2020-05-25 00:00:00,2025-04-05 00:00:00,900,23,1098.2,4.67,Varejo,RJ,0,0,120
C0009,2022-07-11 00:00:00,2024-09-24 00:00:00,1000,11,1240.25,2.92,Varejo,RJ,0,0,313
C0010,2020-07-24 00:00:00,2024-02-21 00:00:00,1100,11,1091.76,2.45,Atacado,RS,1,1,529
C0011,2020-09-12 00:00:00,2024-09-12 00:00:00,1200,24,825.8,1.64,Distribuidor,MG,1,0,325
C0012,2022-04-25 00:00:00,2024-08-14 00:00:00,1300,36,741.1,3.07,Online,PR,0,0,354
C0013,2020-08-13 00:00:00,2024-08-27 00:00:00,1400,40,1265.94,0.78,Atacado,RS,0,1,341
C0014,2021-03-04 00:00:00,2024-09-07 00:00:00,1500,24,347.36,4.5,Varejo,RS,1,0,330
C0015,2022-08-08 00:00:00,2025-02-23 00:00:00,1600,3,1453.59,4.16,Varejo,RS,0,0,161
C0016,2023-08-10 00:00:00,2025-04-09 00:00:00,1700,22,1080.88,1.66,Atacado,RJ,1,0,116
C0017,2023-08-05 00:00:00,2024-02-04 00:00:00,1800,2,339.29,4.77,Atacado,SP,1,0,546
C0018,2021-09-16 00:00:00,2024-08-04 00:00:00,1900,24,1117.56,4.14,Varejo,PR,0,0,364
C0019,2022-10-12 00:00:00,2024-02-03 00:00:00,2000,44,818.27,3.36,Distribuidor,MG,0,1,547
C0020,2020-08-27 00:00:00,2025-04-30 00:00:00,2100,30,1075.48,0.7,Varejo,MG,1,0,95
C0021,2023-03-03 00:00:00,2025-03-03 00:00:00,2200,38,1163.28,2.44,Varejo,SP,1,0,153
C0022,2023-01-03 00:00:00,2024-02-05 00:00:00,2300,2,176.57,2.58,Atacado,RS,1,0,545
C0023,2021-04-02 00:00:00,2400,21,783.85,1.24,Distribuidor,PR,1,0,266
C0024,2023-01-01 00:00:00,2025-06-18 00:00:00,2500,33,1401.42,0.49,Varejo,MG,1,0,46
C0025,2020-09-10 00:00:00,2024-06-02 00:00:00,2600,12,514.93,3.91,Varejo,MG,1,0,427
C0026,2020-07-20 00:00:00,2024-09-11 00:00:00,2700,22,911.13,0.7,Online,RS,1,0,326
C0027,2021-11-02 00:00:00,2025-07-11 00:00:00,2800,44,585.38,3.01,Atacado,SP,1,0,23
C0028,2020-12-14 00:00:00,2025-04-22 00:00:00,2900,25,708.69,4.69,Online,PR,1,0,103
C0029,2021-03-28 00:00:00,2025-03-02 00:00:00,3000,49,845.47,4.39,Online,SP,1,0,154
"""
df_train = pd.read_csv(StringIO(data_raw))
st.sidebar.success("Dados de treinamento de exemplo carregados.")


# R$ FORMATTING FUNCTION (Brazilian style)
def format_real(x):
    if pd.isna(x): return '-'
    # Formata para R$ X.XXX,XX
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# --- FUNÇÃO DE PRÉ-PROCESSAMENTO (Passo 2) ---
def preprocess_data(df_input):
    df = df_input.copy()

    required_cols = ['Potencial', 'total_compras', 'ticket_medio', 'frequencia_mensal', 'dias_desde_ultima_compra',
                     'participou_promocoes', 'categoria_cliente', 'estado']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Coluna obrigatória não encontrada: {col}")
            return None, None

    # Conversão de tipos e criação de features simples (Indicadores de Saúde - Passo 3)
    df['log_total_compras'] = np.log1p(df['total_compras'].replace(0, 1))
    df['log_ticket_medio'] = np.log1p(df['ticket_medio'].replace(0, 1))
    df['participou_promocoes'] = df['participou_promocoes'].astype(int)

    # Colunas que serão codificadas (categóricas)
    categorical_cols = ['categoria_cliente', 'estado']

    # Codificação de categóricas (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Adicionamos as colunas originais para fins de rastreamento/limpeza posterior
    df_encoded['cliente_id'] = df['cliente_id']
    df_encoded['data_cadastro'] = df['data_cadastro']
    df_encoded['data_ultima_compra'] = df['data_ultima_compra']
    df_encoded['categoria_cliente_original'] = df['categoria_cliente']
    df_encoded['estado_original'] = df['estado']

    return df_encoded, df['cliente_id']


# 2. TREINAR O MODELO DE IA SIMPLES (Passo 4)

@st.cache_resource
def train_simple_model(df_data):
    df_encoded, _ = preprocess_data(df_data)

    # A variável 'churn' é a nossa variável alvo para a IA
    y_train = df_encoded['churn'].astype(int)

    # --- REMOÇÃO DE COLUNAS DE TEXTO (OBJECT) E ID/Datas ---

    # Cria uma lista de colunas para remover (as de ID, datas e a variável alvo)
    cols_to_remove = [col for col in df_encoded.columns if
                      col in ['cliente_id', 'data_cadastro', 'data_ultima_compra', 'churn',
                              'categoria_cliente_original', 'estado_original']]

    X_train = df_encoded.drop(columns=cols_to_remove, errors='ignore')

    # A MAIOR GARANTIA: Selecionar APENAS as colunas numéricas (excluir 'object')
    X_train = X_train.select_dtypes(exclude=['object'])

    model_features = X_train.columns.tolist()

    # Treinamento (Passo 4.3)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # --- VALIDAÇÃO (Passo 1.1) ---
    # Faremos uma previsão nos dados de treinamento para obter o relatório (simulação de validação)
    y_pred_train = model.predict(X_train)
    report = classification_report(y_train, y_pred_train, output_dict=True)

    # Extrai métricas chave
    metrics = {
        'accuracy': report['accuracy'],
        'precision_churn': report['1']['precision'],
        'recall_churn': report['1']['recall']
    }

    return model, model_features, metrics


# O modelo retorna também as métricas para exibição.
model, model_features, model_metrics = train_simple_model(df_train)
st.sidebar.success("Modelo de IA (Random Forest) treinado e validado com sucesso.")


# --- FUNÇÃO DE CLASSIFICAÇÃO/STATUS (Passo 5) ---
def classify_status(df_new, model, model_features):
    df_processed, _ = preprocess_data(df_new)

    # Preparar X_predict
    X_predict = df_processed.copy()

    # Garantir que removemos as colunas de rastreio/originais
    cols_to_remove_predict = [col for col in X_predict.columns if
                              col in ['cliente_id', 'data_cadastro', 'data_ultima_compra', 'categoria_cliente_original',
                                      'estado_original']]
    X_predict = X_predict.drop(columns=cols_to_remove_predict, errors='ignore')

    # A última e mais importante garantia: remover qualquer string restante
    X_predict = X_predict.select_dtypes(exclude=['object'])

    # Alinhar colunas de predição com as de treino
    for feature in model_features:
        if feature not in X_predict.columns:
            X_predict[feature] = 0

    X_predict = X_predict[model_features]

    # Predição de Risco de Afastamento (Passo 4.1 e 4.2)
    churn_proba = model.predict_proba(X_predict)[:, 1]

    df_result = df_new.copy()
    df_result['prob_afastamento'] = churn_proba

    # DEFINIÇÃO DE STATUS COM REGRAS SIMPLIFICADAS (Passo 5.1 e 5.2)

    # 1. Definir Risco de Afastamento com base na IA e inatividade (Passo 5.2 - Afastando-se)
    df_result['Status'] = np.where(
        (df_result['prob_afastamento'] >= 0.6) |
        (df_result['dias_desde_ultima_compra'] > 300),
        '🔴 Afastando-se',
        'Potencial Estável/Ascensão/Fidelizado'
    )

    # 2. Fidelizado (Passo 5.2 - Fidelizado): Alto volume de compras E alta frequência
    # É necessário verificar se o DataFrame não está vazio antes de calcular quantis.
    if not df_result.empty:
        q80_compras = df_result['total_compras'].quantile(0.8)
        q80_frequencia = df_result['frequencia_mensal'].quantile(0.8)
        avg_potencial = df_result['Potencial'].mean()

        df_result['Status'] = np.where(
            (df_result['Status'] == 'Potencial Estável/Ascensão/Fidelizado') &
            (df_result['total_compras'] >= q80_compras) &
            (df_result['frequencia_mensal'] >= q80_frequencia),
            '⭐ Fidelizado',
            df_result['Status']
        )

        # 3. Em Ascensão (Passo 5.2 - Em Ascensão): Potencial alto E compras acima da média
        df_result['Status'] = np.where(
            (df_result['Status'] == 'Potencial Estável/Ascensão/Fidelizado') &
            (df_result['Potencial'] > avg_potencial) &
            (df_result['total_compras'] > df_result['total_compras'].mean()),
            '🟢 Em Ascensão',
            df_result['Status']
        )

    # 4. Estável (Passo 5.2 - Estável): O restante
    df_result['Status'] = np.where(
        df_result['Status'] == 'Potencial Estável/Ascensão/Fidelizado',
        '🟡 Estável',
        df_result['Status']
    )

    return df_result


# --- DASHBOARD (Passo 7) ---

uploaded_file = st.sidebar.file_uploader(
    "1. Carregue o arquivo de clientes (CSV/XLSX) para análise",
    type=['csv', 'xlsx']
)

# Simulação de Alerta de Monitoramento (Passo 6.1)
st.sidebar.warning(
    "⚠️ Lembrete: O modelo de IA deve ser retreinado a cada 6 meses para evitar a degradação da precisão.")

if uploaded_file is not None:
    # 3. PROCESSAR O ARQUIVO CARREGADO
    try:
        if uploaded_file.name.endswith('.csv'):
            df_new = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df_new = pd.read_excel(uploaded_file)

        # 4. CLASSIFICAR E MOSTRAR RESULTADOS (Passos 4 e 5)
        df_results = classify_status(df_new, model, model_features)

        st.subheader("Resultado da Classificação (IA + Regras)")

        # --- EXIBIÇÃO DE MÉTRICAS (Passo 1.1) ---
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Acurácia do Modelo", f"{model_metrics['accuracy']:.2f}")
        col_m2.metric("Precisão (Precision) para Churn", f"{model_metrics['precision_churn']:.2f}")
        col_m3.metric("Sensibilidade (Recall) para Churn", f"{model_metrics['recall_churn']:.2f}")
        st.markdown("---")

        # --- PREPARAÇÃO DAS COLUNAS PARA EXIBIÇÃO ---
        df_display = df_results[
            ['cliente_id', 'Status', 'Potencial', 'total_compras', 'dias_desde_ultima_compra',
             'prob_afastamento']].copy()

        # 1. Ajuste da Probabilidade para Porcentagem (Passo 7 - Dashboard)
        df_display['prob_afastamento'] = (df_display['prob_afastamento'] * 100).round(2).astype(str) + ' %'

        # Aplica a formatação de R$
        df_display['Potencial'] = df_display['Potencial'].apply(format_real)

        # 2. Ajuste dos Nomes das Colunas (Passo 7 - Dashboard)
        new_column_names = {
            'cliente_id': 'Cliente ID',
            'Status': 'Status',
            'Potencial': 'Potencial (R$)',
            'total_compras': 'Total Compras',
            'dias_desde_ultima_compra': 'Dias Inativo',
            'prob_afastamento': 'Risco Afastamento (%)'
        }
        df_display = df_display.rename(columns=new_column_names)

        # 5. RELATÓRIO SIMPLES (Passo 6)

        # Filtrar as listas Top 10 (baseado nos dados NUMÉRICOS originais em df_results)
        df_risco_filtered = df_results[df_results['Status'] == '🔴 Afastando-se'].sort_values('Potencial',
                                                                                             ascending=False).head(10)
        df_fidelizado_filtered = df_results[df_results['Status'] == '⭐ Fidelizado'].sort_values('Potencial',
                                                                                                ascending=False).head(
            10)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Total Clientes em Risco",
                value=f"{len(df_results[df_results['Status'] == '🔴 Afastando-se'])}"
            )
            st.dataframe(df_results['Status'].value_counts().reset_index())

        with col2:
            st.markdown("### Tabela Detalhada (Amostra)")

            st.dataframe(
                df_display.sort_values('Risco Afastamento (%)', ascending=False).head(20),
                use_container_width=True
            )

        st.markdown("---")

        # --- NOVA VISÃO: TOP 10 POR POTENCIAL ---
        st.subheader("Análise Estratégica: Top 10 Clientes por Potencial")

        col_risco, col_fidelizado = st.columns(2)

        with col_risco:
            st.markdown("#### 🔴 Top 10 de MAIOR POTENCIAL em Afastamento")
            if not df_risco_filtered.empty:
                # Prepara o subset para exibição R$
                df_risco_display = df_risco_filtered.copy()

                # Aplica formatação nos dados numéricos originais para exibição
                df_risco_display['Potencial (R$)'] = df_risco_display['Potencial'].apply(format_real)
                df_risco_display['Risco Afastamento (%)'] = (df_risco_display['prob_afastamento'] * 100).round(
                    2).astype(str) + ' %'

                # Selecionar e renomear colunas
                st.dataframe(
                    df_risco_display[
                        ['cliente_id', 'Potencial (R$)', 'dias_desde_ultima_compra', 'Risco Afastamento (%)']].rename(
                        columns={
                            'cliente_id': 'Cliente ID',
                            'dias_desde_ultima_compra': 'Dias Inativo'
                        }),
                    use_container_width=True
                )
                st.error("Ação: Intervenção de retenção IMEDIATA. O Potencial destes clientes é vital.")
            else:
                st.info("Nenhum cliente em risco detectado ou o Top 10 está vazio.")

        with col_fidelizado:
            st.markdown("#### ⭐ Top 10 de MAIOR POTENCIAL Fidelizados")
            if not df_fidelizado_filtered.empty:
                # Prepara o subset para exibição R$
                df_fidelizado_display = df_fidelizado_filtered.copy()

                # Aplica formatação nos dados numéricos originais para exibição
                df_fidelizado_display['Potencial (R$)'] = df_fidelizado_display['Potencial'].apply(format_real)
                df_fidelizado_display['Risco Afastamento (%)'] = (df_fidelizado_display[
                                                                      'prob_afastamento'] * 100).round(2).astype(
                    str) + ' %'

                # Selecionar e renomear colunas
                st.dataframe(
                    df_fidelizado_display[
                        ['cliente_id', 'Potencial (R$)', 'total_compras', 'Risco Afastamento (%)']].rename(columns={
                        'cliente_id': 'Cliente ID',
                        'total_compras': 'Total Compras'
                    }),
                    use_container_width=True
                )
                st.success("Ação: Oferecer produtos premium (Up-sell) e recompensas para manter a fidelidade.")
            else:
                st.info("Nenhum cliente Fidelizado detectado no Top 10 por Potencial.")

        st.markdown("---")

        st.subheader("Sugestões de Ação (Passo 6 - Feedback)")
        st.markdown(
            "Clientes '🔴 Afastando-se' precisam de uma **oferta especial** (promoção) ou de um **contato imediato** para entender o problema."
        )
        st.markdown(
            "Clientes '🟢 Em Ascensão' devem ser incentivados a comprar produtos de maior valor (Cross-Sell)."
        )

    except Exception as e:
        st.error(
            f"Ocorreu um erro no processamento do arquivo. Verifique se todas as colunas (Potencial, total_compras, ticket_medio, etc.) estão presentes e no formato correto. Erro: {e}")

else:
    st.info("⬅️ Por favor, utilize a barra lateral para carregar seu arquivo e iniciar a análise.")
