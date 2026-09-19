import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
import zipfile
import io


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Proyecto Final | Online Shoppers",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    [data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARGA DEL DATASET
# ============================================================

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip"
)


@st.cache_data
def cargar_datos():

    with urllib.request.urlopen(DATA_URL) as response:
        contenido = response.read()

    with zipfile.ZipFile(io.BytesIO(contenido)) as archivo_zip:

        archivos_csv = [
            nombre
            for nombre in archivo_zip.namelist()
            if nombre.lower().endswith(".csv")
        ]

        if not archivos_csv:
            raise FileNotFoundError(
                "No se encontró ningún archivo CSV dentro del ZIP."
            )

        with archivo_zip.open(archivos_csv[0]) as archivo_csv:
            df_original = pd.read_csv(archivo_csv)

    duplicados = int(df_original.duplicated().sum())

    df_limpio = (
        df_original
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_original, df_limpio, duplicados


try:
    df_original, df, duplicados = cargar_datos()

except Exception as error:
    st.error("No fue posible cargar el dataset desde UCI.")
    st.write("Detalle:", str(error))
    st.stop()


# ============================================================
# VARIABLES
# ============================================================

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
    "Feb",
    "Mar",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]


# ============================================================
# FUNCIONES
# ============================================================

def tasa_conversion(data):
    if len(data) == 0:
        return 0.0

    return float(data["Revenue"].mean() * 100)


def preparar_revenue(data):

    temporal = data.copy()

    temporal["Resultado"] = temporal["Revenue"].map(
        {
            False: "No compró",
            True: "Compró"
        }
    )

    return temporal


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 Proyecto Final")

st.sidebar.caption(
    "Online Shoppers Purchasing Intention"
)

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

st.sidebar.subheader("🎛️ Filtros")


# ============================================================
# FILTRO MES
# ============================================================

meses_disponibles = [
    mes
    for mes in orden_meses
    if mes in df["Month"].unique()
]

meses_seleccionados = st.sidebar.multiselect(
    "Mes",
    options=meses_disponibles,
    default=meses_disponibles
)


# ============================================================
# FILTRO VISITANTE
# ============================================================

visitantes_disponibles = sorted(
    df["VisitorType"]
    .dropna()
    .unique()
    .tolist()
)

visitantes_seleccionados = st.sidebar.multiselect(
    "Tipo de visitante",
    options=visitantes_disponibles,
    default=visitantes_disponibles
)


# ============================================================
# FILTRO WEEKEND
# ============================================================

tipo_dia = st.sidebar.selectbox(
    "Tipo de día",
    [
        "Todos",
        "Entre semana",
        "Fin de semana"
    ]
)


# ============================================================
# APLICAR FILTROS
# ============================================================

df_filtrado = df.copy()

df_filtrado = df_filtrado[
    df_filtrado["Month"].isin(meses_seleccionados)
]

df_filtrado = df_filtrado[
    df_filtrado["VisitorType"].isin(
        visitantes_seleccionados
    )
]

if tipo_dia == "Entre semana":

    df_filtrado = df_filtrado[
        df_filtrado["Weekend"] == False
    ]

elif tipo_dia == "Fin de semana":

    df_filtrado = df_filtrado[
        df_filtrado["Weekend"] == True
    ]


st.sidebar.divider()

st.sidebar.metric(
    "Sesiones seleccionadas",
    f"{len(df_filtrado):,}"
)

st.sidebar.caption(
    "Los filtros modifican las visualizaciones del dashboard."
)


if len(df_filtrado) == 0:

    st.warning(
        "No existen registros con los filtros seleccionados. "
        "Cambia los filtros de la barra lateral."
    )

    st.stop()


# ============================================================
# INICIO
# ============================================================

if pagina == "🏠 Inicio":

    st.title("🛒 Online Shoppers Purchasing Intention")

    st.markdown(
        "### Análisis interactivo del comportamiento de compra en e-commerce"
    )

    st.write(
        """
        Nuestro equipo eligió el dataset **Online Shoppers Purchasing
        Intention Dataset**, el cual contiene información sobre usuarios
        que navegaron dentro de una página web y el comportamiento que
        tuvieron durante su sesión, teniendo como resultado final si
        compraron o no compraron.

        A partir de las variables disponibles, buscamos identificar
        cuáles están más relacionadas con la decisión de compra y
        utilizarlas para construir posteriormente un modelo que pueda
        estimar si una sesión terminará en una compra.

        Conocer estas variables permite entender qué aspectos de la
        experiencia del usuario tienen mayor relación con el proceso
        de compra y puede ayudar a detectar oportunidades de mejora.
        """
    )

    st.info(
        """
        **Hipótesis de negocio**

        Nuestra hipótesis es que existen características dentro del
        comportamiento de navegación que permiten distinguir a los
        usuarios que terminan comprando de aquellos que no, y que esta
        información puede utilizarse para mejorar la toma de decisiones
        dentro de una empresa de e-commerce.
        """
    )

    st.divider()

    total = len(df_filtrado)
    compras = int(df_filtrado["Revenue"].sum())
    no_compras = total - compras
    conversion = tasa_conversion(df_filtrado)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Sesiones",
        f"{total:,}"
    )

    col2.metric(
        "Compras",
        f"{compras:,}"
    )

    col3.metric(
        "No compraron",
        f"{no_compras:,}"
    )

    col4.metric(
        "Conversión",
        f"{conversion:.2f}%"
    )

    st.divider()

    col1, col2 = st.columns([1.2, 1])

    with col1:

        temporal = preparar_revenue(df_filtrado)

        resultado = (
            temporal["Resultado"]
            .value_counts()
            .rename_axis("Resultado")
            .reset_index(name="Sesiones")
        )

        fig = px.pie(
            resultado,
            names="Resultado",
            values="Sesiones",
            hole=0.55,
            title="Distribución de Revenue"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Objetivo")

        st.write(
            """
            Analizar el comportamiento de los usuarios,
            identificar las variables más importantes y
            preparar la información para desarrollar un
            modelo de clasificación que permita estimar
            si una sesión terminará en compra.
            """
        )

        st.subheader("Variable objetivo")

        st.code("Revenue", language=None)

        st.write(
            """
            **True:** la sesión terminó en compra.

            **False:** la sesión no terminó en compra.
            """
        )

        st.warning(
            """
            Revenue presenta un desbalance: la mayoría de
            las sesiones no termina en compra. Esto deberá
            considerarse al entrenar y evaluar el modelo.
            """
        )


# ============================================================
# DATASET
# ============================================================

elif pagina == "📊 Dataset":

    st.title("📊 Exploración del Dataset")

    st.write(
        """
        En esta sección se revisa la estructura,
        calidad y contenido de la base de datos.
        """
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
        "Variables",
        df.shape[1]
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👀 Datos",
            "📈 Estadísticas",
            "🧹 Calidad",
            "🏷️ Variables"
        ]
    )

    with tab1:

        st.subheader("Vista previa")

        st.dataframe(
            df_filtrado,
            use_container_width=True,
            height=500
        )

        st.caption(
            f"Mostrando {len(df_filtrado):,} registros "
            "según los filtros seleccionados."
        )

    with tab2:

        st.subheader("Estadísticas descriptivas")

        estadisticas = (
            df_filtrado[variables_numericas]
            .describe()
            .T
        )

        st.dataframe(
            estadisticas,
            use_container_width=True
        )

        st.info(
            """
            De acuerdo con la descripción del dataset,
            existen variables categóricas como
            OperatingSystems, Browser, Region y TrafficType
            representadas mediante números.

            Por este motivo, sus valores no deben
            interpretarse como variables numéricas continuas.
            """
        )

    with tab3:

        st.subheader("Valores nulos")

        nulos = (
            df_original
            .isna()
            .sum()
            .rename_axis("Variable")
            .reset_index(name="Valores nulos")
        )

        fig = px.bar(
            nulos,
            x="Variable",
            y="Valores nulos",
            title="Valores nulos por variable"
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        total_nulos = int(
            df_original.isna().sum().sum()
        )

        if total_nulos == 0:

            st.success(
                "El dataset no presenta valores nulos."
            )

        else:

            st.warning(
                f"Se encontraron {total_nulos:,} valores nulos."
            )

        st.subheader("Registros duplicados")

        st.metric(
            "Duplicados encontrados",
            duplicados
        )

        st.write(
            """
            Debido a que la base no cuenta con un
            identificador único de sesión (ID), no es
            posible distinguir si los registros idénticos
            corresponden a sesiones diferentes o a
            observaciones repetidas.

            Para evitar dar un peso adicional a
            observaciones idénticas dentro del análisis
            y del modelo, se decidió conservar una sola
            ocurrencia de cada registro.
            """
        )

    with tab4:

        st.subheader("Tipos de variables")

        tipos = pd.DataFrame(
            {
                "Variable": df.columns,
                "Tipo": [
                    str(tipo)
                    for tipo in df.dtypes
                ]
            }
        )

        st.dataframe(
            tipos,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Variables numéricas analizadas"
        )

        st.write(
            ", ".join(variables_numericas)
        )


# ============================================================
# EXPLORACIÓN
# ============================================================

elif pagina == "🔎 Exploración":

    st.title("🔎 Análisis Exploratorio")

    st.write(
        """
        Selecciona una variable para analizar
        interactivamente su distribución, posibles
        valores extremos y relación con la decisión
        de compra.
        """
    )

    variable = st.selectbox(
        "Selecciona una variable",
        variables_numericas
    )

    st.divider()

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

        fig_hist = px.histogram(
            df_filtrado,
            x=variable,
            nbins=40,
            title=f"Distribución de {variable}"
        )

        fig_hist.update_layout(
            yaxis_title="Cantidad de sesiones"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    with col2:

        fig_box = px.box(
            df_filtrado,
            y=variable,
            points="outliers",
            title=f"Boxplot de {variable}"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    st.subheader("Compra vs. no compra")

    temporal = preparar_revenue(df_filtrado)

    fig_comparacion = px.box(
        temporal,
        x="Resultado",
        y=variable,
        color="Resultado",
        points="outliers",
        title=f"{variable} según resultado de compra"
    )

    st.plotly_chart(
        fig_comparacion,
        use_container_width=True
    )

    if variable == "PageValues":

        st.info(
            """
            **Interpretación:** PageValues tiende a presentar
            valores considerablemente mayores en las sesiones
            que terminan en compra. Esto indica una asociación
            importante con Revenue.
            """
        )

    elif variable in ["BounceRates", "ExitRates"]:

        st.info(
            """
            **Interpretación:** las sesiones que terminan
            en compra tienden a presentar menores valores
            de abandono. Estas variables podrían aportar
            información relevante para distinguir compradores
            de no compradores.
            """
        )

    elif variable == "ProductRelated":

        st.info(
            """
            **Interpretación:** la cantidad de páginas de
            producto visitadas tiende a ser mayor en las
            sesiones que terminan en compra.
            """
        )

    elif variable == "ProductRelated_Duration":

        st.info(
            """
            **Interpretación:** el tiempo dedicado a páginas
            de producto tiende a ser mayor en sesiones con
            compra. Sin embargo, existen valores extremos
            en ambos grupos.
            """
        )

    elif "Duration" in variable:

        st.info(
            """
            **Interpretación:** la distribución presenta
            sesgo positivo y valores extremos. Estos valores
            no se eliminan automáticamente porque pueden
            representar sesiones reales de mayor duración.
            """
        )

    else:

        st.info(
            """
            Utiliza el histograma y los boxplots para
            comparar el comportamiento de esta variable
            entre sesiones con compra y sin compra.
            """
        )


# ============================================================
# CONVERSIÓN
# ============================================================

elif pagina == "🛍️ Conversión":

    st.title("🛍️ Análisis de Conversión")

    st.write(
        """
        Esta sección permite analizar cómo cambia la
        proporción de compra según diferentes
        características de la sesión.
        """
    )

    total = len(df_filtrado)
    compras = int(df_filtrado["Revenue"].sum())
    no_compras = total - compras
    conversion = tasa_conversion(df_filtrado)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Sesiones",
        f"{total:,}"
    )

    col2.metric(
        "Compras",
        f"{compras:,}"
    )

    col3.metric(
        "No compras",
        f"{no_compras:,}"
    )

    col4.metric(
        "Conversión",
        f"{conversion:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # TIPO DE VISITANTE
    # --------------------------------------------------------

    st.subheader(
        "👤 Conversión por tipo de visitante"
    )

    resumen_visitante = (
        df_filtrado
        .groupby("VisitorType", observed=True)
        .agg(
            Sesiones=("Revenue", "size"),
            Compras=("Revenue", "sum"),
            Conversion=("Revenue", "mean")
        )
        .reset_index()
    )

    resumen_visitante["Conversion"] = (
        resumen_visitante["Conversion"] * 100
    )

    fig_visitante = px.bar(
        resumen_visitante,
        x="VisitorType",
        y="Conversion",
        text_auto=".1f",
        hover_data=[
            "Sesiones",
            "Compras"
        ],
        labels={
            "VisitorType": "Tipo de visitante",
            "Conversion": "Conversión (%)"
        },
        title="Porcentaje de compra por tipo de visitante"
    )

    st.plotly_chart(
        fig_visitante,
        use_container_width=True
    )

    st.info(
        """
        La tasa de compra cambia según el tipo de visitante.
        La categoría Other debe interpretarse con cautela
        debido a que contiene una cantidad mucho menor
        de sesiones.
        """
    )

    # --------------------------------------------------------
    # MES
    # --------------------------------------------------------

    st.subheader(
        "📅 Conversión y sesiones por mes"
    )

    resumen_mes = (
        df_filtrado
        .groupby("Month", observed=True)
        .agg(
            Sesiones=("Revenue", "size"),
            Conversion=("Revenue", "mean")
        )
        .reset_index()
    )

    resumen_mes["Conversion"] = (
        resumen_mes["Conversion"] * 100
    )

    resumen_mes["Month"] = pd.Categorical(
        resumen_mes["Month"],
        categories=orden_meses,
        ordered=True
    )

    resumen_mes = (
        resumen_mes
        .sort_values("Month")
    )

    fig_mes = go.Figure()

    fig_mes.add_trace(
        go.Bar(
            x=resumen_mes["Month"],
            y=resumen_mes["Conversion"],
            name="Conversión (%)"
        )
    )

    fig_mes.add_trace(
        go.Scatter(
            x=resumen_mes["Month"],
            y=resumen_mes["Sesiones"],
            name="Sesiones",
            mode="lines+markers",
            yaxis="y2"
        )
    )

    fig_mes.update_layout(
        title="Conversión y cantidad de sesiones por mes",
        xaxis=dict(
            title="Mes"
        ),
        yaxis=dict(
            title="Conversión (%)"
        ),
        yaxis2=dict(
            title="Cantidad de sesiones",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig_mes,
        use_container_width=True
    )

    st.info(
        """
        El volumen de sesiones no está directamente
        relacionado con una mayor conversión. En el
        dataset completo, noviembre destaca por combinar
        un volumen elevado de sesiones con una tasa de
        conversión alta.
        """
    )

    # --------------------------------------------------------
    # FIN DE SEMANA
    # --------------------------------------------------------

    st.subheader(
        "📆 Conversión según tipo de día"
    )

    resumen_weekend = (
        df_filtrado
        .groupby("Weekend", observed=True)
        .agg(
            Sesiones=("Revenue", "size"),
            Conversion=("Revenue", "mean")
        )
        .reset_index()
    )

    resumen_weekend["Conversion"] = (
        resumen_weekend["Conversion"] * 100
    )

    resumen_weekend["Tipo de día"] = (
        resumen_weekend["Weekend"].map(
            {
                False: "Entre semana",
                True: "Fin de semana"
            }
        )
    )

    fig_weekend = px.bar(
        resumen_weekend,
        x="Tipo de día",
        y="Conversion",
        text_auto=".1f",
        hover_data=["Sesiones"],
        labels={
            "Conversion": "Conversión (%)"
        },
        title="Porcentaje de compra según tipo de día"
    )

    st.plotly_chart(
        fig_weekend,
        use_container_width=True
    )

    st.info(
        """
        Las sesiones realizadas durante fines de semana
        presentan una tasa de compra ligeramente diferente,
        pero la diferencia no es tan marcada como en otras
        variables analizadas.
        """
    )

    # --------------------------------------------------------
    # PAGE VALUES
    # --------------------------------------------------------

    st.subheader(
        "💰 PageValues y conversión"
    )

    temporal = preparar_revenue(df_filtrado)

    fig_page = px.box(
        temporal,
        x="Resultado",
        y="PageValues",
        color="Resultado",
        points="outliers",
        title="PageValues según resultado de compra"
    )

    st.plotly_chart(
        fig_page,
        use_container_width=True
    )

    st.success(
        """
        PageValues presenta una de las diferencias
        visuales más claras entre sesiones con compra
        y sesiones sin compra.
        """
    )


# ============================================================
# CORRELACIONES
# ============================================================

elif pagina == "🔥 Correlaciones":

    st.title("🔥 Análisis de Correlaciones")

    st.write(
        """
        La matriz de correlación permite identificar
        relaciones lineales entre las variables numéricas
        del análisis.
        """
    )

    correlacion = (
        df_filtrado[variables_numericas]
        .corr()
    )

    fig_corr = px.imshow(
        correlacion,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title="Matriz de correlación de variables numéricas"
    )

    fig_corr.update_layout(
        height=750
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Relaciones destacadas"
    )

    corr_bounce_exit = correlacion.loc[
        "BounceRates",
        "ExitRates"
    ]

    corr_product = correlacion.loc[
        "ProductRelated",
        "ProductRelated_Duration"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "BounceRates ↔ ExitRates",
            f"{corr_bounce_exit:.2f}"
        )

        st.write(
            """
            Ambas variables representan comportamientos
            relacionados con el abandono dentro del sitio.
            """
        )

    with col2:

        st.metric(
            "ProductRelated ↔ ProductRelated_Duration",
            f"{corr_product:.2f}"
        )

        st.write(
            """
            La cantidad de páginas de producto visitadas
            presenta una relación elevada con el tiempo
            dedicado a dichas páginas.
            """
        )

    st.warning(
        """
        **Posible multicolinealidad**

        Existen correlaciones elevadas entre variables
        que representan comportamientos similares.
        Esto deberá considerarse durante la selección
        de variables y construcción del modelo.
        """
    )


# ============================================================
# CONCLUSIONES
# ============================================================

elif pagina == "💡 Conclusiones":

    st.title("💡 Conclusiones")

    st.write(
        """
        A partir del análisis exploratorio se identificaron
        diferentes patrones relacionados con el comportamiento
        de compra de los usuarios.
        """
    )

    st.divider()

    st.subheader(
        "1. Revenue presenta un desbalance"
    )

    st.write(
        """
        La mayoría de las sesiones no termina en compra.
        Por esta razón, durante la construcción del modelo
        no será suficiente utilizar únicamente accuracy.
        Será necesario considerar métricas como precision,
        recall, F1-score y ROC-AUC.
        """
    )

    st.subheader(
        "2. PageValues presenta una asociación importante"
    )

    st.write(
        """
        Las sesiones que terminan en compra presentan
        generalmente valores superiores de PageValues.
        Esta variable muestra una diferencia clara entre
        compradores y no compradores.
        """
    )

    st.subheader(
        "3. El abandono se relaciona con la conversión"
    )

    st.write(
        """
        Las sesiones con compra tienden a presentar
        menores valores de BounceRates y ExitRates.
        Estas variables contienen información sobre
        el comportamiento de abandono dentro del sitio.
        """
    )

    st.subheader(
        "4. La interacción con productos aporta información"
    )

    st.write(
        """
        ProductRelated y ProductRelated_Duration tienden
        a presentar valores mayores en las sesiones que
        terminan en compra. Esto sugiere que la interacción
        con páginas de producto está asociada con la
        conversión.
        """
    )

    st.subheader(
        "5. Existen variables correlacionadas"
    )

    st.write(
        """
        BounceRates y ExitRates presentan una correlación
        elevada. También existe una relación importante
        entre ProductRelated y ProductRelated_Duration.

        Estas relaciones deberán considerarse durante la
        selección de variables para el modelo.
        """
    )

    st.subheader(
        "6. Más tráfico no significa necesariamente más conversión"
    )

    st.write(
        """
        El volumen de sesiones cambia entre meses, pero
        una mayor cantidad de tráfico no implica
        automáticamente una mayor tasa de compra.
        """
    )

    st.divider()

    st.success(
        """
        **Resultado de la etapa exploratoria**

        El análisis descriptivo muestra que existen
        características del comportamiento de navegación
        asociadas con la decisión de compra, lo cual es
        consistente con la hipótesis de negocio planteada.

        El siguiente paso será evaluar estas relaciones
        mediante modelos de clasificación.
        """
    )

    st.subheader("Siguiente etapa")

    st.markdown(
        """
        **Modelado de clasificación**

        - Preparación de variables categóricas
        - Separación entre entrenamiento y prueba
        - Tratamiento del desbalance
        - Entrenamiento de modelos
        - Matriz de confusión
        - Precision, Recall y F1-score
        - ROC-AUC
        - Importancia de variables
        - Predictor interactivo de compra
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Proyecto Final · Online Shoppers Purchasing Intention Dataset · "
    "Python + Streamlit + Plotly"
)
