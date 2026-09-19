import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Proyecto Final | Online Shoppers",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ESTILO
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stMetric"] {
    background-color: rgba(120,120,120,0.08);
    border: 1px solid rgba(120,120,120,0.15);
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
}

h1 {
    margin-bottom: 0px;
}

.small-text {
    opacity: 0.7;
    font-size: 14px;
}

.insight {
    padding: 18px;
    border-radius: 10px;
    background-color: rgba(120,120,120,0.08);
    border-left: 4px solid #888;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET
# =========================================================

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00468/online_shoppers_intention.csv"
)


@st.cache_data
def cargar_datos():
    df_original = pd.read_csv(DATA_URL)

    duplicados = df_original.duplicated().sum()

    df_limpio = (
        df_original
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_original, df_limpio, duplicados


try:
    df_original, df, duplicados = cargar_datos()

except Exception:
    st.error(
        "No fue posible cargar el dataset. "
        "Verifica la conexión con la fuente de datos."
    )
    st.stop()


# =========================================================
# VARIABLES
# =========================================================

variables_numericas = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

orden_meses = [
    "Feb", "Mar", "May", "June", "Jul",
    "Aug", "Sep", "Oct", "Nov", "Dec"
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛒 Proyecto Final")
st.sidebar.caption("Online Shoppers Purchasing Intention")

st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "📊 Dataset",
        "🔎 Exploración",
        "🛍️ Conversión",
        "🔥 Correlaciones",
        "💡 Conclusiones"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Filtros globales")

meses_seleccionados = st.sidebar.multiselect(
    "Mes",
    options=orden_meses,
    default=orden_meses
)

tipos_visitante = sorted(
    df["VisitorType"].dropna().unique()
)

visitantes_seleccionados = st.sidebar.multiselect(
    "Tipo de visitante",
    options=tipos_visitante,
    default=tipos_visitante
)

filtro_weekend = st.sidebar.selectbox(
    "Tipo de día",
    [
        "Todos",
        "Entre semana",
        "Fin de semana"
    ]
)


# =========================================================
# APLICAR FILTROS
# =========================================================

df_filtrado = df[
    df["Month"].isin(meses_seleccionados)
    & df["VisitorType"].isin(visitantes_seleccionados)
].copy()

if filtro_weekend == "Entre semana":
    df_filtrado = df_filtrado[
        df_filtrado["Weekend"] == False
    ]

elif filtro_weekend == "Fin de semana":
    df_filtrado = df_filtrado[
        df_filtrado["Weekend"] == True
    ]


st.sidebar.divider()

st.sidebar.metric(
    "Sesiones seleccionadas",
    f"{len(df_filtrado):,}"
)

if len(df_filtrado) == 0:
    st.warning(
        "Los filtros seleccionados no contienen registros. "
        "Modifica los filtros de la barra lateral."
    )
    st.stop()


# =========================================================
# FUNCIONES
# =========================================================

def conversion_rate(data):
    return data["Revenue"].mean() * 100


def grafica_boxplot(variable):
    temp = df_filtrado.copy()

    temp["Resultado"] = temp["Revenue"].map({
        False: "No compró",
        True: "Compró"
    })

    fig = px.box(
        temp,
        x="Resultado",
        y=variable,
        color="Resultado",
        title=f"{variable}: compra vs. no compra",
        points="outliers"
    )

    fig.update_layout(
        legend_title_text="Resultado"
    )

    return fig


# =========================================================
# INICIO
# =========================================================

if pagina == "🏠 Inicio":

    st.title("🛒 Online Shoppers Purchasing Intention")

    st.markdown(
        "### Análisis del comportamiento de compra en e-commerce"
    )

    st.write("""
    Nuestro equipo eligió el dataset **Online Shoppers Purchasing
    Intention Dataset**, el cual contiene información sobre usuarios
    que navegaron dentro de una página web y el comportamiento que
    tuvieron durante su sesión, teniendo como resultado final si
    compraron o no compraron.

    El objetivo del proyecto es identificar cuáles variables están
    más relacionadas con la decisión de compra y utilizarlas para
    comprender mejor el comportamiento de los usuarios y apoyar
    posteriormente la construcción de un modelo de clasificación.
    """)

    st.info("""
    **Hipótesis de negocio**

    Existen características dentro del comportamiento de navegación
    que permiten distinguir a los usuarios que terminan comprando de
    aquellos que no, y esta información puede utilizarse para mejorar
    la toma de decisiones dentro de una empresa de e-commerce.
    """)

    st.divider()

    st.subheader("Resumen general")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Sesiones",
        f"{len(df_filtrado):,}"
    )

    col2.metric(
        "Compras",
        f"{df_filtrado['Revenue'].sum():,}"
    )

    col3.metric(
        "Conversión",
        f"{conversion_rate(df_filtrado):.1f}%"
    )

    col4.metric(
        "Variables",
        df.shape[1]
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        resultados = (
            df_filtrado["Revenue"]
            .value_counts()
            .rename(index={
                False: "No compró",
                True: "Compró"
            })
            .reset_index()
        )

        resultados.columns = [
            "Resultado",
            "Sesiones"
        ]

        fig = px.pie(
            resultados,
            names="Resultado",
            values="Sesiones",
            hole=0.55,
            title="Distribución de la variable objetivo"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Objetivo")

        st.write("""
        Analizar el comportamiento de navegación de los usuarios,
        identificar las variables que presentan mayor relación con
        la conversión y utilizar esta información para apoyar
        decisiones orientadas a mejorar la experiencia del cliente
        y optimizar recursos.
        """)

        st.subheader("Variable objetivo")

        st.code("Revenue", language=None)

        st.write("""
        - **True:** la sesión terminó en compra.
        - **False:** la sesión no terminó en compra.
        """)


# =========================================================
# DATASET
# =========================================================

elif pagina == "📊 Dataset":

    st.title("📊 Exploración del Dataset")

    st.write(
        "En esta sección se revisa la estructura y calidad "
        "de la información utilizada."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Registros originales",
        f"{len(df_original):,}"
    )

    col2.metric(
        "Duplicados",
        f"{duplicados:,}"
    )

    col3.metric(
        "Registros finales",
        f"{len(df):,}"
    )

    col4.metric(
        "Valores nulos",
        f"{df.isna().sum().sum():,}"
    )

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "Vista de datos",
        "Estadísticas",
        "Calidad de datos"
    ])

    with tab1:

        st.subheader("Datos")

        st.dataframe(
            df_filtrado,
            use_container_width=True,
            height=500
        )

        st.caption(
            f"Mostrando {len(df_filtrado):,} sesiones "
            "de acuerdo con los filtros seleccionados."
        )

    with tab2:

        st.subheader(
            "Estadísticas descriptivas"
        )

        st.dataframe(
            df_filtrado[
                variables_numericas
            ].describe().T,
            use_container_width=True
        )

        st.info("""
        Algunas variables categóricas como OperatingSystems,
        Browser, Region y TrafficType se encuentran representadas
        mediante valores numéricos, por lo que no deben interpretarse
        directamente como variables numéricas continuas.
        """)

    with tab3:

        st.subheader(
            "Valores nulos por variable"
        )

        nulos = (
            df.isna()
            .sum()
            .reset_index()
        )

        nulos.columns = [
            "Variable",
            "Valores nulos"
        ]

        fig = px.bar(
            nulos,
            x="Variable",
            y="Valores nulos"
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if nulos["Valores nulos"].sum() == 0:
            st.success(
                "El dataset no presenta valores nulos."
            )

        st.subheader("Registros duplicados")

        st.write(
            f"Se encontraron **{duplicados} registros duplicados** "
            "en la base original."
        )

        st.write("""
        Debido a que el dataset no cuenta con un identificador único
        de sesión, no es posible distinguir si corresponden a sesiones
        diferentes con exactamente las mismas características o a
        registros repetidos.

        Para evitar otorgar un peso adicional a observaciones
        idénticas, se decidió conservar una sola ocurrencia de cada
        registro.
        """)


# =========================================================
# EXPLORACIÓN
# =========================================================

elif pagina == "🔎 Exploración":

    st.title("🔎 Análisis Exploratorio")

    st.write("""
    Selecciona una variable para explorar su distribución,
    detectar valores extremos y comparar su comportamiento entre
    sesiones que terminaron o no en compra.
    """)

    variable = st.selectbox(
        "Variable numérica",
        variables_numericas
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Promedio",
        f"{df_filtrado[variable].mean():.2f}"
    )

    col2.metric(
        "Mediana",
        f"{df_filtrado[variable].median():.2f}"
    )

    col3.metric(
        "Máximo",
        f"{df_filtrado[variable].max():.2f}"
    )

    col4.metric(
        "Desv. estándar",
        f"{df_filtrado[variable].std():.2f}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df_filtrado,
            x=variable,
            nbins=40,
            title=f"Distribución de {variable}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df_filtrado,
            y=variable,
            points="outliers",
            title=f"Valores atípicos de {variable}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Comparación con la decisión de compra"
    )

    st.plotly_chart(
        grafica_boxplot(variable),
        use_container_width=True
    )

    if variable == "PageValues":

        st.info("""
        **Interpretación:** PageValues presenta diferencias claras
        entre sesiones con compra y sin compra. Las sesiones que
        terminan en conversión tienden a presentar valores mayores.
        """)

    elif variable in ["BounceRates", "ExitRates"]:

        st.info("""
        **Interpretación:** las sesiones que terminan en compra
        tienden a presentar menores valores de abandono, lo cual
        sugiere una relación entre una mayor permanencia dentro del
        sitio y la conversión
