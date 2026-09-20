# ============================================================
# PROYECTO FINAL
# ONLINE SHOPPERS PURCHASING INTENTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.request
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Online Shoppers | Proyecto Final",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background-color: rgba(120,120,120,0.08);
        border: 1px solid rgba(120,120,120,0.18);
        padding: 16px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        font-size: 27px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET
# ============================================================

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip"
)


@st.cache_data
def cargar_datos():

    with urllib.request.urlopen(DATA_URL) as response:
        contenido = response.read()

    with zipfile.ZipFile(
        io.BytesIO(contenido)
    ) as archivo_zip:

        archivos_csv = [
            nombre
            for nombre in archivo_zip.namelist()
            if nombre.lower().endswith(".csv")
        ]

        if not archivos_csv:
            raise FileNotFoundError(
                "No se encontró un archivo CSV."
            )

        with archivo_zip.open(
            archivos_csv[0]
        ) as archivo:

            df_original = pd.read_csv(
                archivo
            )

    # --------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------

    duplicados = int(
        df_original.duplicated().sum()
    )

    df_limpio = (
        df_original
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return (
        df_original,
        df_limpio,
        duplicados
    )


try:

    (
        df_original,
        df,
        duplicados
    ) = cargar_datos()

except Exception as error:

    st.error(
        "No fue posible cargar el dataset."
    )

    st.write(error)

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


variables_numericas_modelo = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay"
]


variables_categoricas = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
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

    return float(
        data["Revenue"].mean()
        * 100
    )


def agregar_resultado(data):

    temp = data.copy()

    temp["Resultado"] = (
        temp["Revenue"]
        .map(
            {
                False: "No compra",
                True: "Compra"
            }
        )
    )

    return temp


def paso_slider(serie):

    rango = (
        float(serie.max())
        - float(serie.min())
    )

    if rango <= 1:

        return 0.001

    elif rango <= 10:

        return 0.1

    elif rango <= 100:

        return 1.0

    else:

        return max(
            1.0,
            round(
                rango / 500,
                2
            )
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

@st.cache_resource
def entrenar_modelos(data):

    df_modelo = data.copy()

    # --------------------------------------------------------
    # PAGEVALUES FUERA DEL MODELO
    # --------------------------------------------------------

    df_modelo.drop(
        columns=["PageValues"],
        inplace=True
    )

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X = df_modelo.drop(
        "Revenue",
        axis=1
    )

    y = (
        df_modelo["Revenue"]
        .astype(int)
    )

    # --------------------------------------------------------
    # ONE HOT ENCODING
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=variables_categoricas,
        dtype=int
    )

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        X_train.copy()
    )

    X_test_scaled = (
        X_test.copy()
    )

    X_train_scaled[
        variables_numericas_modelo
    ] = scaler.fit_transform(
        X_train[
            variables_numericas_modelo
        ]
    )

    X_test_scaled[
        variables_numericas_modelo
    ] = scaler.transform(
        X_test[
            variables_numericas_modelo
        ]
    )

    # --------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------

    modelos = {

        "Regresión Logística":
            LogisticRegression(
                solver="saga",
                random_state=42
            ),

        "Árbol de Decisión":
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=7,
                random_state=42
            ),

        "k-NN":
            KNeighborsClassifier(
                n_neighbors=25,
                metric="euclidean"
            ),

        "SVM":
            SVC(
                random_state=42,
                kernel="poly",
                C=0.1
            ),

        "Random Forest":
            RandomForestClassifier(
                class_weight="balanced",
                max_depth=5,
                min_samples_leaf=5,
                random_state=42
            )
    }

    resultados = []

    matrices = {}

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    for nombre, modelo in modelos.items():

        modelo.fit(
            X_train_scaled,
            y_train
        )

        pred = modelo.predict(
            X_test_scaled
        )

        resultados.append(
            {
                "Modelo":
                    nombre,

                "Accuracy":
                    accuracy_score(
                        y_test,
                        pred
                    ),

                "F1 Macro":
                    f1_score(
                        y_test,
                        pred,
                        average="macro",
                        zero_division=0
                    ),

                "Precision Compra":
                    precision_score(
                        y_test,
                        pred,
                        pos_label=1,
                        zero_division=0
                    ),

                "Recall Compra":
                    recall_score(
                        y_test,
                        pred,
                        pos_label=1,
                        zero_division=0
                    ),

                "Precision No compra":
                    precision_score(
                        y_test,
                        pred,
                        pos_label=0,
                        zero_division=0
                    ),

                "Recall No compra":
                    recall_score(
                        y_test,
                        pred,
                        pos_label=0,
                        zero_division=0
                    )
            }
        )

        matrices[nombre] = (
            confusion_matrix(
                y_test,
                pred,
                labels=[0, 1]
            )
        )

    resultados = pd.DataFrame(
        resultados
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    random_forest = (
        modelos["Random Forest"]
    )

    importancia = pd.DataFrame(
        {
            "Variable":
                X.columns,

            "Importancia":
                random_forest.feature_importances_
        }
    )

    importancia = (
        importancia
        .sort_values(
            "Importancia",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return {

        "X":
            X,

        "X_train":
            X_train,

        "X_test":
            X_test,

        "X_train_scaled":
            X_train_scaled,

        "X_test_scaled":
            X_test_scaled,

        "y_train":
            y_train,

        "y_test":
            y_test,

        "modelos":
            modelos,

        "resultados":
            resultados,

        "matrices":
            matrices,

        "importancia":
            importancia,

        "scaler":
            scaler
    }


with st.spinner(
    "Preparando modelos..."
):

    ml = entrenar_modelos(
        df
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛒 PROYECTO FINAL"
)

st.sidebar.caption(
    "Online Shoppers Purchasing Intention"
)

st.sidebar.divider()


pagina = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "📊 Dataset limpio",
        "🔎 Exploración interactiva",
        "🎛️ Comportamiento vs Revenue",
        "📅 Mayo vs Noviembre",
        "👥 Nuevos vs recurrentes",
        "🛍️ Conversión",
        "🔥 Relaciones entre variables",
        "⚙️ Preparación del modelo",
        "🤖 Modelos de Machine Learning",
        "📊 Comparación de modelos",
        "🌲 Random Forest",
        "📈 Variables importantes",
        "💼 Propuesta de negocio y conclusiones"
    ]
)


st.sidebar.divider()


# ============================================================
# FILTROS
# ============================================================

st.sidebar.subheader(
    "Filtros exploratorios"
)


meses_disponibles = [
    mes
    for mes in orden_meses
    if mes in df[
        "Month"
    ].unique()
]


meses_seleccionados = (
    st.sidebar.multiselect(
        "Mes",
        meses_disponibles,
        default=meses_disponibles
    )
)


visitantes_disponibles = sorted(
    df[
        "VisitorType"
    ]
    .dropna()
    .unique()
    .tolist()
)


visitantes_seleccionados = (
    st.sidebar.multiselect(
        "Tipo de visitante",
        visitantes_disponibles,
        default=visitantes_disponibles
    )
)


dia_seleccionado = (
    st.sidebar.selectbox(
        "Tipo de día",
        [
            "Todos",
            "Entre semana",
            "Fin de semana"
        ]
    )
)


# ============================================================
# APLICAR FILTROS
# ============================================================

df_filtrado = df[
    df[
        "Month"
    ].isin(
        meses_seleccionados
    )
    &
    df[
        "VisitorType"
    ].isin(
        visitantes_seleccionados
    )
].copy()


if (
    dia_seleccionado
    == "Entre semana"
):

    df_filtrado = df_filtrado[
        df_filtrado[
            "Weekend"
        ] == False
    ]


elif (
    dia_seleccionado
    == "Fin de semana"
):

    df_filtrado = df_filtrado[
        df_filtrado[
            "Weekend"
        ] == True
    ]


st.sidebar.metric(
    "Sesiones visibles",
    f"{len(df_filtrado):,}"
)


st.sidebar.caption(
    """
    Estos filtros afectan únicamente
    el análisis exploratorio.

    Los modelos utilizan siempre
    el dataset limpio completo.
    """
)


if len(df_filtrado) == 0:

    st.warning(
        """
        No existen sesiones con
        los filtros seleccionados.
        """
    )

    st.stop()


# ============================================================
# INICIO
# ============================================================

if pagina == "🏠 Inicio":

    st.title(
        "🛒 Comportamiento de usuarios en e-commerce"
    )

    st.subheader(
        "Online Shoppers Purchasing Intention"
    )

    st.write(
        """
        Este dashboard transforma el análisis realizado
        en Google Colab en una herramienta interactiva
        para estudiar el comportamiento de los usuarios
        durante el proceso de compra.
        """
    )

    st.info(
        """
        ### Pregunta de negocio

        ¿Qué características distinguen a las sesiones
        que terminan en compra y cómo podemos utilizar
        esos patrones para mejorar la conversión?
        """
    )

    st.success(
        """
        ### Propuesta

        Analizar qué hace diferente a noviembre,
        estudiar qué patrones podrían orientar mejoras
        durante mayo y comparar el comportamiento
        de visitantes nuevos y recurrentes.

        Posteriormente, utilizar Random Forest para
        identificar señales de compra generadas durante
        la navegación.
        """
    )

    total = len(
        df_filtrado
    )

    compras = int(
        df_filtrado[
            "Revenue"
        ].sum()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

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
        f"{total-compras:,}"
    )

    col4.metric(
        "Conversión",
        f"{tasa_conversion(df_filtrado):.2f}%"
    )

    st.divider()

    temp = agregar_resultado(
        df_filtrado
    )

    col1, col2 = st.columns(
        [1.2, 1]
    )

    with col1:

        revenue_conteo = (
            temp[
                "Resultado"
            ]
            .value_counts()
            .rename_axis(
                "Resultado"
            )
            .reset_index(
                name="Sesiones"
            )
        )

        fig = px.pie(
            revenue_conteo,
            names="Resultado",
            values="Sesiones",
            hole=0.55,
            title="Distribución de Revenue"
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Variable objetivo"
        )

        st.code(
            "Revenue"
        )

        st.write(
            """
            **True:** sesión con compra.

            **False:** sesión sin compra.
            """
        )

        st.warning(
            """
            Revenue está desbalanceado.

            Por eso Accuracy no debe
            utilizarse como única métrica.
            """
        )

    st.divider()

    st.subheader(
        "Flujo del proyecto"
    )

    st.write(
        """
        1. Limpieza del dataset

        2. EDA

        3. Exploración interactiva

        4. Análisis Mayo vs Noviembre

        5. Nuevos vs recurrentes

        6. Modelos supervisados

        7. Random Forest

        8. Propuesta de negocio
        """
    )

    st.subheader(
        "👥 Integrantes"
    )

    st.write(
        """
        - Daniela Pineda Fuentes
        - Diego Hurtado Baker
        - Santiago Becerril Martínez
        - Daniela Jayani Magdaleno Rojas
        - Mariana Ruiz Ramírez
        - Christofer Muñiz Martínez
        """
    )


# ============================================================
# DATASET LIMPIO
# ============================================================

elif pagina == "📊 Dataset limpio":

    st.title(
        "📊 Dataset limpio"
    )

    st.write(
        """
        Se utiliza el mismo proceso
        de limpieza desarrollado
        en Google Colab.
        """
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Filas originales",
        f"{len(df_original):,}"
    )

    col2.metric(
        "Duplicados",
        f"{duplicados:,}"
    )

    col3.metric(
        "Filas limpias",
        f"{len(df):,}"
    )

    col4.metric(
        "Columnas",
        df.shape[1]
    )

    st.success(
        f"""
        Se eliminaron **{duplicados} registros duplicados**.

        El análisis utiliza
        **{len(df):,} registros limpios**.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Vista de datos",
            "Estadísticas",
            "Calidad",
            "Metadatos"
        ]
    )

    with tab1:

        st.dataframe(
            df_filtrado,
            use_container_width=True,
            height=500
        )

    with tab2:

        st.dataframe(
            df_filtrado[
                variables_numericas
            ].describe().T,
            use_container_width=True
        )

    with tab3:

        col1, col2 = (
            st.columns(2)
        )

        col1.metric(
            "Valores nulos",
            int(
                df_original
                .isna()
                .sum()
                .sum()
            )
        )

        col2.metric(
            "Duplicados originales",
            duplicados
        )

    with tab4:

        tipos = pd.DataFrame(
            {
                "Variable":
                    df.columns,

                "Tipo":
                    [
                        str(tipo)
                        for tipo
                        in df.dtypes
                    ]
            }
        )

        st.dataframe(
            tipos,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        """
        Fuente: UCI Machine Learning Repository —
        Online Shoppers Purchasing Intention Dataset.
        """
    )
    # ============================================================
# EXPLORACIÓN INTERACTIVA
# ============================================================

elif pagina == "🔎 Exploración interactiva":

    st.title(
        "🔎 Exploración interactiva"
    )

    st.write(
        """
        Selecciona una variable y modifica su rango
        para observar cómo cambia Revenue utilizando
        únicamente sesiones reales del dataset limpio.
        """
    )

    variable = st.selectbox(
        "Variable a analizar",
        variables_numericas
    )

    minimo = float(
        df_filtrado[variable].min()
    )

    maximo = float(
        df_filtrado[variable].max()
    )

    if maximo > minimo:

        rango = st.slider(
            f"Rango de {variable}",
            min_value=minimo,
            max_value=maximo,
            value=(minimo, maximo),
            step=paso_slider(
                df_filtrado[variable]
            )
        )

        df_exploracion = df_filtrado[
            df_filtrado[variable].between(
                rango[0],
                rango[1]
            )
        ].copy()

    else:

        df_exploracion = (
            df_filtrado.copy()
        )

    if len(df_exploracion) == 0:

        st.warning(
            """
            No existen sesiones
            en ese rango.
            """
        )

        st.stop()

    conversion_actual = (
        tasa_conversion(
            df_exploracion
        )
    )

    conversion_base = (
        tasa_conversion(
            df_filtrado
        )
    )

    diferencia = (
        conversion_actual
        - conversion_base
    )

    compras = int(
        df_exploracion[
            "Revenue"
        ].sum()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Sesiones reales",
        f"{len(df_exploracion):,}"
    )

    col2.metric(
        "Compras",
        f"{compras:,}"
    )

    col3.metric(
        "No compras",
        f"{len(df_exploracion)-compras:,}"
    )

    col4.metric(
        "Conversión",
        f"{conversion_actual:.2f}%",
        delta=f"{diferencia:+.2f} pp"
    )

    temp = agregar_resultado(
        df_exploracion
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        fig = px.histogram(
            temp,
            x=variable,
            color="Resultado",
            nbins=40,
            barmode="overlay",
            title=(
                f"Distribución de {variable}"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            temp,
            x="Resultado",
            y=variable,
            color="Resultado",
            points="outliers",
            title=(
                f"{variable}: Compra vs No compra"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "📈 Variable vs tasa de compra"
    )

    datos_rangos = (
        df_filtrado[
            [variable, "Revenue"]
        ]
        .dropna()
        .copy()
    )

    try:

        datos_rangos["Rango"] = (
            pd.qcut(
                datos_rangos[variable],
                q=8,
                duplicates="drop"
            )
        )

    except ValueError:

        datos_rangos["Rango"] = (
            pd.cut(
                datos_rangos[variable],
                bins=8,
                duplicates="drop"
            )
        )

    conversion_rangos = (
        datos_rangos
        .groupby(
            "Rango",
            observed=True
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),
            Conversion=(
                "Revenue",
                "mean"
            )
        )
        .reset_index()
    )

    conversion_rangos[
        "Conversion"
    ] *= 100

    conversion_rangos[
        "Rango"
    ] = (
        conversion_rangos[
            "Rango"
        ]
        .astype(str)
    )

    fig = px.bar(
        conversion_rangos,
        x="Rango",
        y="Conversion",
        hover_data=[
            "Sesiones"
        ],
        text_auto=".1f",
        labels={
            "Conversion":
                "Tasa de compra (%)",
            "Rango":
                variable
        },
        title=(
            f"Tasa de compra según {variable}"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if variable == "ProductRelated":

        st.info(
            """
            ProductRelated representa la cantidad de páginas
            relacionadas con productos visitadas durante
            una sesión.

            Se utiliza como aproximación a la interacción
            con productos, pero no representa literalmente
            el número de clicks.
            """
        )

    if variable == "PageValues":

        st.warning(
            """
            PageValues se utiliza en el EDA, pero se elimina
            posteriormente del Machine Learning por posible
            riesgo de target leakage.
            """
        )


# ============================================================
# COMPORTAMIENTO VS REVENUE
# ============================================================

elif pagina == "🎛️ Comportamiento vs Revenue":

    st.title(
        "🎛️ Comportamiento vs Revenue"
    )

    st.write(
        """
        Modifica diferentes características de navegación
        y observa qué ocurre con Revenue.
        """
    )

    st.success(
        """
        Los sliders filtran sesiones reales.

        No generan usuarios artificiales.
        """
    )

    prod_min = int(
        df_filtrado[
            "ProductRelated"
        ].min()
    )

    prod_max = int(
        df_filtrado[
            "ProductRelated"
        ].max()
    )

    prod_dur_min = float(
        df_filtrado[
            "ProductRelated_Duration"
        ].min()
    )

    prod_dur_max = float(
        df_filtrado[
            "ProductRelated_Duration"
        ].max()
    )

    bounce_min = float(
        df_filtrado[
            "BounceRates"
        ].min()
    )

    bounce_max = float(
        df_filtrado[
            "BounceRates"
        ].max()
    )

    exit_min = float(
        df_filtrado[
            "ExitRates"
        ].min()
    )

    exit_max = float(
        df_filtrado[
            "ExitRates"
        ].max()
    )

    admin_min = int(
        df_filtrado[
            "Administrative"
        ].min()
    )

    admin_max = int(
        df_filtrado[
            "Administrative"
        ].max()
    )

    info_min = int(
        df_filtrado[
            "Informational"
        ].min()
    )

    info_max = int(
        df_filtrado[
            "Informational"
        ].max()
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        rango_prod = st.slider(
            "Páginas de producto visitadas",
            min_value=prod_min,
            max_value=prod_max,
            value=(
                prod_min,
                prod_max
            )
        )

        rango_prod_dur = st.slider(
            "Tiempo en páginas de producto",
            min_value=prod_dur_min,
            max_value=prod_dur_max,
            value=(
                prod_dur_min,
                prod_dur_max
            ),
            step=paso_slider(
                df_filtrado[
                    "ProductRelated_Duration"
                ]
            )
        )

        rango_admin = st.slider(
            "Páginas administrativas",
            min_value=admin_min,
            max_value=admin_max,
            value=(
                admin_min,
                admin_max
            )
        )

    with col2:

        rango_bounce = st.slider(
            "Bounce Rate",
            min_value=bounce_min,
            max_value=bounce_max,
            value=(
                bounce_min,
                bounce_max
            ),
            step=0.001
        )

        rango_exit = st.slider(
            "Exit Rate",
            min_value=exit_min,
            max_value=exit_max,
            value=(
                exit_min,
                exit_max
            ),
            step=0.001
        )

        rango_info = st.slider(
            "Páginas informativas",
            min_value=info_min,
            max_value=info_max,
            value=(
                info_min,
                info_max
            )
        )

    simulacion = df_filtrado[
        df_filtrado[
            "ProductRelated"
        ].between(
            rango_prod[0],
            rango_prod[1]
        )
        &
        df_filtrado[
            "ProductRelated_Duration"
        ].between(
            rango_prod_dur[0],
            rango_prod_dur[1]
        )
        &
        df_filtrado[
            "BounceRates"
        ].between(
            rango_bounce[0],
            rango_bounce[1]
        )
        &
        df_filtrado[
            "ExitRates"
        ].between(
            rango_exit[0],
            rango_exit[1]
        )
        &
        df_filtrado[
            "Administrative"
        ].between(
            rango_admin[0],
            rango_admin[1]
        )
        &
        df_filtrado[
            "Informational"
        ].between(
            rango_info[0],
            rango_info[1]
        )
    ].copy()

    st.divider()

    if len(simulacion) == 0:

        st.warning(
            """
            No existen sesiones reales que cumplan
            todos los parámetros.

            Amplía alguno de los rangos.
            """
        )

    else:

        conversion_original = (
            tasa_conversion(
                df_filtrado
            )
        )

        conversion_sim = (
            tasa_conversion(
                simulacion
            )
        )

        cambio = (
            conversion_sim
            - conversion_original
        )

        compras = int(
            simulacion[
                "Revenue"
            ].sum()
        )

        no_compras = (
            len(simulacion)
            - compras
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Sesiones",
            f"{len(simulacion):,}"
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
            "Tasa de compra",
            f"{conversion_sim:.2f}%",
            delta=f"{cambio:+.2f} pp"
        )

        temp = agregar_resultado(
            simulacion
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            conteo = (
                temp[
                    "Resultado"
                ]
                .value_counts()
                .rename_axis(
                    "Resultado"
                )
                .reset_index(
                    name="Sesiones"
                )
            )

            fig = px.pie(
                conteo,
                names="Resultado",
                values="Sesiones",
                hole=0.5,
                title=(
                    "Revenue de sesiones seleccionadas"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.histogram(
                temp,
                x="ProductRelated",
                color="Resultado",
                barmode="overlay",
                nbins=35,
                title=(
                    "Interacción con productos"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        fig = px.scatter(
            temp,
            x="ProductRelated",
            y="ProductRelated_Duration",
            color="Resultado",
            opacity=0.60,
            hover_data=[
                "BounceRates",
                "ExitRates",
                "VisitorType",
                "Month"
            ],
            title=(
                "Páginas de producto vs duración"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.warning(
            """
            Estas relaciones son asociaciones observadas
            en los datos. No demuestran causalidad.
            """
        )


# ============================================================
# MAYO VS NOVIEMBRE
# ============================================================

elif pagina == "📅 Mayo vs Noviembre":

    st.title(
        "📅 Mayo vs Noviembre"
    )

    st.write(
        """
        Analizamos qué hace diferente a noviembre respecto
        a mayo y qué representaría mejorar el Conversion Rate
        de mayo.
        """
    )

    # Se utiliza el dataset limpio completo
    # para que la comparación no dependa de los filtros laterales.

    df_mayo = df[
        df["Month"] == "May"
    ].copy()

    df_noviembre = df[
        df["Month"] == "Nov"
    ].copy()

    # ========================================================
    # MÉTRICAS
    # ========================================================

    sesiones_mayo = len(
        df_mayo
    )

    sesiones_noviembre = len(
        df_noviembre
    )

    compras_mayo = int(
        df_mayo[
            "Revenue"
        ].sum()
    )

    compras_noviembre = int(
        df_noviembre[
            "Revenue"
        ].sum()
    )

    no_compras_mayo = (
        sesiones_mayo
        - compras_mayo
    )

    conversion_mayo = (
        tasa_conversion(
            df_mayo
        )
    )

    conversion_noviembre = (
        tasa_conversion(
            df_noviembre
        )
    )

    # ========================================================
    # MAYO VS NOVIEMBRE
    # ========================================================

    st.subheader(
        "📊 Situación observada"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.markdown(
            "### Mayo"
        )

        a, b, c = (
            st.columns(3)
        )

        a.metric(
            "Sesiones",
            f"{sesiones_mayo:,}"
        )

        b.metric(
            "Compras",
            f"{compras_mayo:,}"
        )

        c.metric(
            "Conversión",
            f"{conversion_mayo:.2f}%"
        )

    with col2:

        st.markdown(
            "### Noviembre"
        )

        a, b, c = (
            st.columns(3)
        )

        a.metric(
            "Sesiones",
            f"{sesiones_noviembre:,}"
        )

        b.metric(
            "Compras",
            f"{compras_noviembre:,}"
        )

        c.metric(
            "Conversión",
            f"{conversion_noviembre:.2f}%"
        )

    comparacion_conversion = pd.DataFrame(
        {
            "Mes": [
                "Mayo",
                "Noviembre"
            ],
            "Conversion Rate": [
                conversion_mayo,
                conversion_noviembre
            ],
            "Sesiones": [
                sesiones_mayo,
                sesiones_noviembre
            ],
            "Compras": [
                compras_mayo,
                compras_noviembre
            ]
        }
    )

    fig = px.bar(
        comparacion_conversion,
        x="Mes",
        y="Conversion Rate",
        text_auto=".2f",
        hover_data=[
            "Sesiones",
            "Compras"
        ],
        labels={
            "Conversion Rate":
                "Conversion Rate (%)"
        },
        title=(
            "Conversion Rate: Mayo vs Noviembre"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # QUÉ HACE DIFERENTE A NOVIEMBRE
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 ¿Qué hace diferente a noviembre?"
    )

    st.write(
        """
        Selecciona una variable para comparar
        su comportamiento en mayo y noviembre.
        """
    )

    variables_comparacion = [
        "ProductRelated",
        "ProductRelated_Duration",
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues"
    ]

    variable_comparar = (
        st.selectbox(
            "Variable a comparar",
            variables_comparacion
        )
    )

    media_mayo = (
        df_mayo[
            variable_comparar
        ].mean()
    )

    media_noviembre = (
        df_noviembre[
            variable_comparar
        ].mean()
    )

    diferencia = (
        media_noviembre
        - media_mayo
    )

    if media_mayo != 0:

        cambio_relativo = (
            (
                media_noviembre
                / media_mayo
            )
            - 1
        ) * 100

    else:

        cambio_relativo = 0

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Promedio Mayo",
        f"{media_mayo:,.3f}"
    )

    col2.metric(
        "Promedio Noviembre",
        f"{media_noviembre:,.3f}"
    )

    col3.metric(
        "Diferencia",
        f"{diferencia:+,.3f}"
    )

    col4.metric(
        "Cambio relativo",
        f"{cambio_relativo:+.1f}%"
    )

    comparacion_variable = pd.concat(
        [
            df_mayo[
                [variable_comparar]
            ].assign(
                Mes="Mayo"
            ),

            df_noviembre[
                [variable_comparar]
            ].assign(
                Mes="Noviembre"
            )
        ],
        ignore_index=True
    )

    fig = px.box(
        comparacion_variable,
        x="Mes",
        y=variable_comparar,
        color="Mes",
        points="outliers",
        title=(
            f"{variable_comparar}: "
            "Mayo vs Noviembre"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # TABLA GENERAL MAYO VS NOVIEMBRE
    # ========================================================

    st.subheader(
        "📋 Comparación general"
    )

    resumen_variables = []

    for variable in variables_comparacion:

        valor_mayo = (
            df_mayo[
                variable
            ].mean()
        )

        valor_noviembre = (
            df_noviembre[
                variable
            ].mean()
        )

        diferencia_variable = (
            valor_noviembre
            - valor_mayo
        )

        if valor_mayo != 0:

            cambio_pct = (
                (
                    valor_noviembre
                    / valor_mayo
                    - 1
                )
                * 100
            )

        else:

            cambio_pct = np.nan

        resumen_variables.append(
            {
                "Variable":
                    variable,

                "Mayo":
                    valor_mayo,

                "Noviembre":
                    valor_noviembre,

                "Diferencia":
                    diferencia_variable,

                "Cambio %":
                    cambio_pct
            }
        )

    resumen_variables = (
        pd.DataFrame(
            resumen_variables
        )
    )

    st.dataframe(
        resumen_variables.style.format(
            {
                "Mayo":
                    "{:.3f}",

                "Noviembre":
                    "{:.3f}",

                "Diferencia":
                    "{:+.3f}",

                "Cambio %":
                    "{:+.1f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # VISITOR TYPE MAYO VS NOVIEMBRE
    # ========================================================

    st.divider()

    st.subheader(
        "👥 Tipo de visitante"
    )

    st.write(
        """
        Revisamos si parte de la diferencia entre ambos meses
        está relacionada con la composición de los visitantes.
        """
    )

    visitantes = pd.concat(
        [
            df_mayo.assign(
                Mes_completo="Mayo"
            ),

            df_noviembre.assign(
                Mes_completo="Noviembre"
            )
        ],
        ignore_index=True
    )

    resumen_visitantes = (
        visitantes
        .groupby(
            [
                "Mes_completo",
                "VisitorType"
            ],
            observed=True
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),

            Compras=(
                "Revenue",
                "sum"
            ),

            Conversion=(
                "Revenue",
                "mean"
            )
        )
        .reset_index()
    )

    resumen_visitantes[
        "Conversion"
    ] *= 100

    st.dataframe(
        resumen_visitantes,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        resumen_visitantes,
        x="VisitorType",
        y="Conversion",
        color="Mes_completo",
        barmode="group",
        text_auto=".1f",
        hover_data=[
            "Sesiones",
            "Compras"
        ],
        labels={
            "VisitorType":
                "Tipo de visitante",

            "Conversion":
                "Conversion Rate (%)",

            "Mes_completo":
                "Mes"
        },
        title=(
            "Conversión por tipo de visitante"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # SIMULADOR DE CONVERSION RATE EN MAYO
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Simulador de mejora en Mayo"
    )

    st.write(
        """
        Mueve el slider para proyectar cuántas sesiones
        adicionales terminarían en compra si mejoramos
        el Conversion Rate de mayo.
        """
    )

    st.info(
        """
        En esta simulación el número de sesiones de mayo
        permanece constante.

        Solamente se modifica el Conversion Rate.
        """
    )

    minimo_slider = float(
        np.floor(
            conversion_mayo
        )
    )

    maximo_slider = float(
        max(
            40,
            np.ceil(
                conversion_noviembre
                + 10
            )
        )
    )

    valor_inicial = float(
        round(
            max(
                conversion_mayo,
                conversion_noviembre
            ),
            1
        )
    )

    valor_inicial = min(
        valor_inicial,
        maximo_slider
    )

    conversion_objetivo = (
        st.slider(
            "Conversion Rate objetivo para Mayo",
            min_value=minimo_slider,
            max_value=maximo_slider,
            value=valor_inicial,
            step=0.5,
            format="%.1f%%"
        )
    )

    # ========================================================
    # PROYECCIÓN DE REVENUE
    # ========================================================

    compras_proyectadas = int(
        round(
            sesiones_mayo
            * (
                conversion_objetivo
                / 100
            )
        )
    )

    compras_adicionales = (
        compras_proyectadas
        - compras_mayo
    )

    no_compras_proyectadas = (
        sesiones_mayo
        - compras_proyectadas
    )

    mejora_pp = (
        conversion_objetivo
        - conversion_mayo
    )

    if conversion_mayo > 0:

        mejora_relativa = (
            (
                conversion_objetivo
                / conversion_mayo
            )
            - 1
        ) * 100

    else:

        mejora_relativa = 0

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Conversión actual",
        f"{conversion_mayo:.2f}%"
    )

    col2.metric(
        "Conversión objetivo",
        f"{conversion_objetivo:.2f}%",
        delta=f"{mejora_pp:+.2f} pp"
    )

    col3.metric(
        "Revenue=True proyectado",
        f"{compras_proyectadas:,}"
    )

    col4.metric(
        "Compras adicionales",
        f"{compras_adicionales:+,}"
    )

    st.caption(
        f"""
        Mejora relativa del Conversion Rate:
        {mejora_relativa:.1f}%.
        """
    )

    st.write(
        f"""
        Con las **{sesiones_mayo:,} sesiones reales de mayo**,
        pasar de una conversión de **{conversion_mayo:.2f}%**
        a **{conversion_objetivo:.2f}%** representaría
        aproximadamente:

        ### {compras_proyectadas:,} sesiones con Revenue=True

        frente a:

        ### {compras_mayo:,} compras observadas actualmente.

        Esto representa:

        ### {compras_adicionales:+,} compras adicionales.
        """
    )

    # ========================================================
    # ACTUAL VS PROYECTADO
    # ========================================================

    escenario_revenue = pd.DataFrame(
        {
            "Escenario": [
                "Mayo actual",
                "Mayo proyectado"
            ],

            "Compra": [
                compras_mayo,
                compras_proyectadas
            ],

            "No compra": [
                no_compras_mayo,
                no_compras_proyectadas
            ]
        }
    )

    escenario_largo = (
        escenario_revenue
        .melt(
            id_vars="Escenario",
            var_name="Revenue",
            value_name="Sesiones"
        )
    )

    fig = px.bar(
        escenario_largo,
        x="Escenario",
        y="Sesiones",
        color="Revenue",
        barmode="stack",
        text_auto=True,
        title=(
            "Cambio proyectado en Revenue de Mayo"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # MAYO ACTUAL / OBJETIVO / NOVIEMBRE
    # ========================================================

    escenarios_conversion = (
        pd.DataFrame(
            {
                "Escenario": [
                    "Mayo actual",
                    "Mayo objetivo",
                    "Noviembre real"
                ],

                "Conversion Rate": [
                    conversion_mayo,
                    conversion_objetivo,
                    conversion_noviembre
                ]
            }
        )
    )

    fig = px.bar(
        escenarios_conversion,
        x="Escenario",
        y="Conversion Rate",
        text_auto=".2f",
        labels={
            "Conversion Rate":
                "Conversion Rate (%)"
        },
        title=(
            "Mayo actual vs objetivo vs Noviembre"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # CURVA PROYECTADA
    # ========================================================

    st.subheader(
        "📈 Impacto según Conversion Rate"
    )

    tasas = np.arange(
        minimo_slider,
        maximo_slider + 0.5,
        0.5
    )

    curva = pd.DataFrame(
        {
            "Conversion Rate":
                tasas
        }
    )

    curva[
        "Revenue=True proyectado"
    ] = (
        sesiones_mayo
        * (
            curva[
                "Conversion Rate"
            ]
            / 100
        )
    ).round()

    fig = px.line(
        curva,
        x="Conversion Rate",
        y="Revenue=True proyectado",
        markers=True,
        labels={
            "Conversion Rate":
                "Conversion Rate (%)",

            "Revenue=True proyectado":
                "Sesiones con compra"
        },
        title=(
            "Compras proyectadas según Conversion Rate"
        )
    )

    fig.add_vline(
        x=conversion_mayo,
        line_dash="dash",
        annotation_text="Mayo actual"
    )

    fig.add_vline(
        x=conversion_noviembre,
        line_dash="dash",
        annotation_text="Noviembre"
    )

    fig.add_vline(
        x=conversion_objetivo,
        line_dash="dot",
        annotation_text="Objetivo"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # MAYO CON TASA DE NOVIEMBRE
    # ========================================================

    st.divider()

    st.subheader(
        "🚀 Mayo con el Conversion Rate de Noviembre"
    )

    compras_nivel_noviembre = int(
        round(
            sesiones_mayo
            * (
                conversion_noviembre
                / 100
            )
        )
    )

    compras_extra_noviembre = (
        compras_nivel_noviembre
        - compras_mayo
    )

    brecha = (
        conversion_noviembre
        - conversion_mayo
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Brecha de conversión",
        f"{brecha:+.2f} pp"
    )

    col2.metric(
        "Revenue=True proyectado",
        f"{compras_nivel_noviembre:,}"
    )

    col3.metric(
        "Compras adicionales",
        f"{compras_extra_noviembre:+,}"
    )

    st.write(
        f"""
        Si las **{sesiones_mayo:,} sesiones de mayo**
        hubieran tenido la tasa de conversión observada
        en noviembre (**{conversion_noviembre:.2f}%**),
        el escenario correspondería aproximadamente a:

        **{compras_nivel_noviembre:,} compras**

        o

        **{compras_extra_noviembre:+,} compras adicionales**
        respecto a mayo.
        """
    )

    # ========================================================
    # IMPACTO MONETARIO OPCIONAL
    # ========================================================

    st.divider()

    st.subheader(
        "💰 Impacto monetario estimado"
    )

    st.write(
        """
        El dataset no contiene el valor monetario
        de las compras.

        Si conocemos el ticket promedio,
        podemos estimar el impacto económico.
        """
    )

    ticket_promedio = (
        st.number_input(
            "Ticket promedio por compra ($)",
            min_value=0.0,
            value=0.0,
            step=100.0
        )
    )

    if ticket_promedio > 0:

        ingreso_actual = (
            compras_mayo
            * ticket_promedio
        )

        ingreso_proyectado = (
            compras_proyectadas
            * ticket_promedio
        )

        ingreso_extra = (
            compras_adicionales
            * ticket_promedio
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Ingreso actual estimado",
            f"${ingreso_actual:,.2f}"
        )

        col2.metric(
            "Ingreso proyectado",
            f"${ingreso_proyectado:,.2f}"
        )

        col3.metric(
            "Ingreso adicional",
            f"${ingreso_extra:,.2f}"
        )

    else:

        st.caption(
            """
            Introduce un ticket promedio
            para calcular el impacto económico.
            """
        )

    st.warning(
        """
        Esta sección representa una simulación
        matemática de escenario.

        No demuestra que aumentar una variable
        específica vaya a causar automáticamente
        este incremento.

        La comparación Mayo vs Noviembre y
        Random Forest ayudan a investigar qué
        características están asociadas con
        mayores tasas de compra.
        """
    )


# ============================================================
# NUEVOS VS RECURRENTES
# ============================================================

elif pagina == "👥 Nuevos vs recurrentes":

    st.title(
        "👥 Nuevos vs recurrentes"
    )

    st.write(
        """
        Analizamos las diferencias observadas
        entre visitantes nuevos y recurrentes.
        """
    )

    resumen_visitantes = (
        df
        .groupby(
            "VisitorType",
            observed=True
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),

            Compras=(
                "Revenue",
                "sum"
            ),

            Conversion=(
                "Revenue",
                "mean"
            ),

            ProductRelated=(
                "ProductRelated",
                "mean"
            ),

            ProductRelated_Duration=(
                "ProductRelated_Duration",
                "mean"
            ),

            BounceRates=(
                "BounceRates",
                "mean"
            ),

            ExitRates=(
                "ExitRates",
                "mean"
            )
        )
        .reset_index()
    )

    resumen_visitantes[
        "Conversion"
    ] *= 100

    total_sesiones = (
        resumen_visitantes[
            "Sesiones"
        ].sum()
    )

    resumen_visitantes[
        "% de sesiones"
    ] = (
        resumen_visitantes[
            "Sesiones"
        ]
        / total_sesiones
        * 100
    )

    st.dataframe(
        resumen_visitantes,
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        fig = px.bar(
            resumen_visitantes,
            x="VisitorType",
            y="Sesiones",
            text_auto=True,
            title=(
                "Sesiones por tipo de visitante"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            resumen_visitantes,
            x="VisitorType",
            y="Conversion",
            text_auto=".1f",
            labels={
                "Conversion":
                    "Conversion Rate (%)"
            },
            title=(
                "Conversión por tipo de visitante"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "📆 Comportamiento mensual"
    )

    visitor_month = (
        df
        .groupby(
            [
                "Month",
                "VisitorType"
            ],
            observed=True
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),

            Conversion=(
                "Revenue",
                "mean"
            )
        )
        .reset_index()
    )

    visitor_month[
        "Conversion"
    ] *= 100

    visitor_month[
        "Month"
    ] = pd.Categorical(
        visitor_month[
            "Month"
        ],
        categories=orden_meses,
        ordered=True
    )

    visitor_month = (
        visitor_month
        .sort_values(
            "Month"
        )
    )

    fig = px.line(
        visitor_month,
        x="Month",
        y="Conversion",
        color="VisitorType",
        markers=True,
        hover_data=[
            "Sesiones"
        ],
        labels={
            "Conversion":
                "Conversion Rate (%)"
        },
        title=(
            "Conversión mensual por tipo de visitante"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        El dataset permite comparar sesiones
        de usuarios nuevos y recurrentes.

        Sin embargo, no permite determinar
        exactamente por qué una persona
        específica decidió no regresar,
        porque no existe seguimiento
        individual longitudinal.
        """
    )
    # ============================================================
# CONVERSIÓN
# ============================================================

elif pagina == "🛍️ Conversión":

    st.title(
        "🛍️ Análisis de conversión"
    )

    conversion = (
        tasa_conversion(
            df_filtrado
        )
    )

    compras = int(
        df_filtrado[
            "Revenue"
        ].sum()
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Sesiones",
        f"{len(df_filtrado):,}"
    )

    col2.metric(
        "Compras",
        f"{compras:,}"
    )

    col3.metric(
        "Conversión",
        f"{conversion:.2f}%"
    )

    # --------------------------------------------------------
    # VISITOR TYPE
    # --------------------------------------------------------

    st.subheader(
        "Conversión por tipo de visitante"
    )

    visitante = (
        df_filtrado
        .groupby(
            "VisitorType",
            observed=True
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),
            Conversion=(
                "Revenue",
                "mean"
            )
        )
        .reset_index()
    )

    visitante[
        "Conversion"
    ] *= 100

    fig = px.bar(
        visitante,
        x="VisitorType",
        y="Conversion",
        hover_data=[
            "Sesiones"
        ],
        text_auto=".1f",
        labels={
            "VisitorType":
                "Tipo de visitante",
            "Conversion":
                "Conversión (%)"
        },
        title=(
            "Conversión por tipo de visitante"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # MES
    # --------------------------------------------------------

    st.subheader(
        "Conversión por mes"
    )

    mensual = (
        df_filtrado
        .groupby(
            "Month",
            observed=True
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),
            Conversion=(
                "Revenue",
                "mean"
            )
        )
        .reset_index()
    )

    mensual[
        "Conversion"
    ] *= 100

    mensual[
        "Month"
    ] = pd.Categorical(
        mensual[
            "Month"
        ],
        categories=orden_meses,
        ordered=True
    )

    mensual = (
        mensual
        .sort_values(
            "Month"
        )
    )

    fig = px.line(
        mensual,
        x="Month",
        y="Conversion",
        markers=True,
        hover_data=[
            "Sesiones"
        ],
        labels={
            "Month":
                "Mes",
            "Conversion":
                "Conversión (%)"
        },
        title=(
            "Conversion Rate mensual"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # WEEKEND
    # --------------------------------------------------------

    st.subheader(
        "Fin de semana vs entre semana"
    )

    weekend = (
        df_filtrado
        .groupby(
            "Weekend"
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size"
            ),
            Conversion=(
                "Revenue",
                "mean"
            )
        )
        .reset_index()
    )

    weekend[
        "Conversion"
    ] *= 100

    weekend[
        "Tipo de día"
    ] = weekend[
        "Weekend"
    ].map(
        {
            False:
                "Entre semana",
            True:
                "Fin de semana"
        }
    )

    fig = px.bar(
        weekend,
        x="Tipo de día",
        y="Conversion",
        hover_data=[
            "Sesiones"
        ],
        text_auto=".1f",
        labels={
            "Conversion":
                "Conversión (%)"
        },
        title=(
            "Conversión según tipo de día"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RELACIONES ENTRE VARIABLES
# ============================================================

elif pagina == "🔥 Relaciones entre variables":

    st.title(
        "🔥 Relaciones entre variables"
    )

    st.write(
        """
        Selecciona dos variables para analizar su relación
        y observar cómo se distribuyen las sesiones
        con compra y sin compra.
        """
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        variable_x = (
            st.selectbox(
                "Variable X",
                variables_numericas,
                index=4
            )
        )

    with col2:

        variable_y = (
            st.selectbox(
                "Variable Y",
                variables_numericas,
                index=5
            )
        )

    correlacion_xy = (
        df_filtrado[
            [
                variable_x,
                variable_y
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    st.metric(
        "Correlación",
        f"{correlacion_xy:.3f}"
    )

    temp = agregar_resultado(
        df_filtrado
    )

    fig = px.scatter(
        temp,
        x=variable_x,
        y=variable_y,
        color="Resultado",
        opacity=0.55,
        hover_data=[
            "Month",
            "VisitorType"
        ],
        title=(
            f"{variable_x} vs {variable_y}"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Matriz de correlaciones"
    )

    matriz = (
        df_filtrado[
            variables_numericas
        ]
        .corr()
    )

    fig = px.imshow(
        matriz,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=(
            "Correlaciones de variables numéricas"
        )
    )

    fig.update_layout(
        height=720
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "BounceRates ↔ ExitRates",
        f"""
        {matriz.loc[
            "BounceRates",
            "ExitRates"
        ]:.2f}
        """
    )

    col2.metric(
        "ProductRelated ↔ Duration",
        f"""
        {matriz.loc[
            "ProductRelated",
            "ProductRelated_Duration"
        ]:.2f}
        """
    )


# ============================================================
# PREPARACIÓN DEL MODELO
# ============================================================

elif pagina == "⚙️ Preparación del modelo":

    st.title(
        "⚙️ Preparación del modelo"
    )

    st.subheader(
        "1. Dataset limpio"
    )

    st.write(
        f"""
        Registros originales:
        **{len(df_original):,}**

        Duplicados eliminados:
        **{duplicados:,}**

        Registros utilizados:
        **{len(df):,}**
        """
    )

    st.subheader(
        "2. Eliminación de PageValues"
    )

    st.warning(
        """
        PageValues se mantiene durante el análisis
        exploratorio, pero se elimina de los modelos
        debido al posible riesgo de target leakage.
        """
    )

    st.subheader(
        "3. Variables categóricas"
    )

    st.write(
        """
        Se aplica One-Hot Encoding mediante
        `pd.get_dummies()` a:
        """
    )

    st.code(
        """
Month
OperatingSystems
Browser
Region
TrafficType
VisitorType
Weekend
        """
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Predictores finales",
        ml[
            "X"
        ].shape[1]
    )

    col2.metric(
        "Train",
        f"{len(ml['X_train']):,}"
    )

    col3.metric(
        "Test",
        f"{len(ml['X_test']):,}"
    )

    st.subheader(
        "4. Train / Test"
    )

    st.write(
        """
        Se utiliza una división:

        **80% entrenamiento**

        **20% prueba**

        con:

        `random_state = 42`

        `stratify = y`
        """
    )

    st.subheader(
        "5. StandardScaler"
    )

    st.write(
        """
        Las variables numéricas son estandarizadas
        utilizando únicamente la información
        aprendida del conjunto de entrenamiento.
        """
    )

    st.subheader(
        "6. Modelos"
    )

    st.write(
        """
        Se comparan cinco modelos de clasificación:

        - Regresión Logística
        - Árbol de Decisión
        - k-NN
        - SVM
        - Random Forest
        """
    )


# ============================================================
# MODELOS DE MACHINE LEARNING
# ============================================================

elif pagina == "🤖 Modelos de Machine Learning":

    st.title(
        "🤖 Modelos de Machine Learning"
    )

    st.write(
        """
        Selecciona un modelo para revisar sus métricas
        y matriz de confusión.
        """
    )

    modelo_seleccionado = (
        st.selectbox(
            "Selecciona un modelo",
            ml[
                "resultados"
            ][
                "Modelo"
            ].tolist()
        )
    )

    fila = (
        ml[
            "resultados"
        ]
        .set_index(
            "Modelo"
        )
        .loc[
            modelo_seleccionado
        ]
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Accuracy",
        f"{fila['Accuracy']*100:.2f}%"
    )

    col2.metric(
        "F1 Macro",
        f"{fila['F1 Macro']:.2f}"
    )

    col3.metric(
        "Precision Compra",
        f"{fila['Precision Compra']*100:.2f}%"
    )

    col4.metric(
        "Recall Compra",
        f"{fila['Recall Compra']*100:.2f}%"
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Precision No compra",
        f"{fila['Precision No compra']*100:.2f}%"
    )

    col2.metric(
        "Recall No compra",
        f"{fila['Recall No compra']*100:.2f}%"
    )

    matriz_modelo = (
        ml[
            "matrices"
        ][
            modelo_seleccionado
        ]
    )

    matriz_pct = (
        matriz_modelo
        / matriz_modelo.sum()
        * 100
    )

    fig = px.imshow(
        matriz_pct,
        text_auto=".1f",
        x=[
            "Predice No compra",
            "Predice Compra"
        ],
        y=[
            "Real No compra",
            "Real Compra"
        ],
        labels={
            "x":
                "Predicción",
            "y":
                "Valor real",
            "color":
                "% del test"
        },
        title=(
            f"Matriz de confusión — "
            f"{modelo_seleccionado}"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if (
        modelo_seleccionado
        == "SVM"
    ):

        st.warning(
            """
            SVM muestra un fuerte sesgo hacia
            la clase mayoritaria.

            Por ello, una Accuracy alta puede
            resultar engañosa en un problema
            con clases desbalanceadas.
            """
        )

    if (
        modelo_seleccionado
        == "Random Forest"
    ):

        st.info(
            """
            Random Forest detecta una proporción
            mayor de las sesiones que terminan
            en compra, aunque también genera
            más falsos positivos.
            """
        )


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

elif pagina == "📊 Comparación de modelos":

    st.title(
        "📊 Comparación de modelos"
    )

    resultados = (
        ml[
            "resultados"
        ].copy()
    )

    tabla = (
        resultados.copy()
    )

    columnas_metricas = [
        "Accuracy",
        "F1 Macro",
        "Precision Compra",
        "Recall Compra",
        "Precision No compra",
        "Recall No compra"
    ]

    for columna in columnas_metricas:

        tabla[
            columna
        ] = (
            tabla[
                columna
            ]
            * 100
        ).round(2)

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Accuracy y F1 Macro"
    )

    comparacion = (
        resultados.melt(
            id_vars="Modelo",
            value_vars=[
                "Accuracy",
                "F1 Macro"
            ],
            var_name="Métrica",
            value_name="Valor"
        )
    )

    fig = px.bar(
        comparacion,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        text_auto=".2f",
        title=(
            "Comparación general de modelos"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "🎯 Detección de compradores"
    )

    fig = px.bar(
        resultados,
        x="Modelo",
        y="Recall Compra",
        text_auto=".2f",
        title=(
            "Recall para la clase Compra"
        )
    )

    fig.update_layout(
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        """
        Debido al desbalance de Revenue,
        Accuracy no debe utilizarse
        como único criterio de evaluación.
        """
    )


# ============================================================
# RANDOM FOREST
# ============================================================

elif pagina == "🌲 Random Forest":

    st.title(
        "🌲 Random Forest"
    )

    rf = (
        ml[
            "resultados"
        ]
        .set_index(
            "Modelo"
        )
        .loc[
            "Random Forest"
        ]
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Accuracy",
        f"{rf['Accuracy']*100:.2f}%"
    )

    col2.metric(
        "F1 Macro",
        f"{rf['F1 Macro']:.2f}"
    )

    col3.metric(
        "Recall Compra",
        f"{rf['Recall Compra']*100:.2f}%"
    )

    col4.metric(
        "Precision Compra",
        f"{rf['Precision Compra']*100:.2f}%"
    )

    st.subheader(
        "¿Por qué nos interesa este modelo?"
    )

    st.write(
        f"""
        Bajo el criterio de priorizar la detección
        de compradores potenciales, Random Forest
        identifica aproximadamente:

        **{rf['Recall Compra']*100:.1f}% de las compras reales**

        presentes en el conjunto de prueba.
        """
    )

    st.write(
        f"""
        Sin embargo, su Precision para Compra es
        aproximadamente:

        **{rf['Precision Compra']*100:.1f}%**

        Esto indica que también genera una cantidad
        relevante de falsos positivos.
        """
    )

    matriz_rf = (
        ml[
            "matrices"
        ][
            "Random Forest"
        ]
    )

    matriz_rf_pct = (
        matriz_rf
        / matriz_rf.sum()
        * 100
    )

    fig = px.imshow(
        matriz_rf_pct,
        text_auto=".1f",
        x=[
            "Predice No compra",
            "Predice Compra"
        ],
        y=[
            "Real No compra",
            "Real Compra"
        ],
        labels={
            "x":
                "Predicción",
            "y":
                "Valor real",
            "color":
                "% del test"
        },
        title=(
            "Matriz de confusión — Random Forest"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        Debido a su Precision, este tipo de modelo
        podría ser más útil para intervenciones
        de bajo costo, como recomendaciones,
        personalización de contenido o mensajes,
        que para descuentos costosos aplicados
        automáticamente.
        """
    )


# ============================================================
# VARIABLES IMPORTANTES
# ============================================================

elif pagina == "📈 Variables importantes":

    st.title(
        "📈 Variables importantes"
    )

    st.write(
        """
        Estas son las variables que tienen mayor peso
        dentro de las decisiones del Random Forest.
        """
    )

    importancia = (
        ml[
            "importancia"
        ]
        .head(12)
        .copy()
    )

    fig = px.bar(
        importancia.sort_values(
            "Importancia"
        ),
        x="Importancia",
        y="Variable",
        orientation="h",
        title=(
            "Top 12 variables — Random Forest"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Top 5 variables"
    )

    top5 = (
        importancia.head(5)
    )

    for posicion, fila in enumerate(
        top5.itertuples(),
        start=1
    ):

        st.write(
            f"""
            **{posicion}. {fila.Variable}**
            — importancia: {fila.Importancia:.4f}
            """
        )

    st.warning(
        """
        Feature importance indica qué variables
        fueron relevantes para las decisiones
        del modelo.

        No significa que esas variables causen
        directamente una compra.
        """
    )


# ============================================================
# PROPUESTA DE NEGOCIO Y CONCLUSIONES
# ============================================================

elif pagina == "💼 Propuesta de negocio y conclusiones":

    st.title(
        "💼 Propuesta de negocio y conclusiones"
    )

    rf = (
        ml[
            "resultados"
        ]
        .set_index(
            "Modelo"
        )
        .loc[
            "Random Forest"
        ]
    )

    df_mayo_final = df[
        df[
            "Month"
        ] == "May"
    ]

    df_noviembre_final = df[
        df[
            "Month"
        ] == "Nov"
    ]

    conversion_mayo_final = (
        tasa_conversion(
            df_mayo_final
        )
    )

    conversion_noviembre_final = (
        tasa_conversion(
            df_noviembre_final
        )
    )

    st.subheader(
        "🎯 Propuesta de negocio"
    )

    st.write(
        """
        La propuesta consiste en analizar qué características
        hacen diferente a noviembre y utilizar esos hallazgos
        como referencia para diseñar acciones que permitan
        mejorar la conversión durante mayo.
        """
    )

    st.subheader(
        "1. Entender qué hace diferente a Noviembre"
    )

    st.write(
        f"""
        La tasa de conversión observada es:

        **Mayo:** {conversion_mayo_final:.2f}%

        **Noviembre:** {conversion_noviembre_final:.2f}%

        El dashboard permite comparar variables como:

        - ProductRelated
        - ProductRelated_Duration
        - BounceRates
        - ExitRates
        - Administrative
        - Informational

        para investigar qué diferencias de comportamiento
        existen entre ambos meses.
        """
    )

    st.subheader(
        "2. Llevar los hallazgos a Mayo"
    )

    st.write(
        """
        El simulador de Mayo permite establecer un
        Conversion Rate objetivo y traducirlo en un
        número aproximado de compras adicionales,
        manteniendo constante el volumen de sesiones.
        """
    )

    st.info(
        """
        Esto ayuda a responder una pregunta de negocio:

        **¿Qué impacto tendría cerrar parte de la brecha
        de conversión entre Mayo y Noviembre?**
        """
    )

    st.subheader(
        "3. Nuevos vs recurrentes"
    )

    st.write(
        """
        VisitorType permite estudiar diferencias en:

        - cantidad de sesiones;
        - tasa de conversión;
        - páginas de producto visitadas;
        - duración de navegación;
        - Bounce Rate;
        - Exit Rate.

        Esto permite identificar oportunidades diferentes
        para adquisición y retención.
        """
    )

    st.subheader(
        "4. Random Forest durante la navegación"
    )

    st.write(
        f"""
        Una vez que el usuario comienza a navegar,
        Random Forest utiliza las señales generadas
        durante la sesión.

        Su Recall para Compra es aproximadamente:

        **{rf['Recall Compra']*100:.1f}%**
        """
    )

    st.write(
        f"""
        Su Precision para Compra es aproximadamente:

        **{rf['Precision Compra']*100:.1f}%**
        """
    )

    st.subheader(
        "5. Estrategia propuesta"
    )

    st.success(
        """
        **Antes o al inicio de la sesión**

        Utilizar variables como Month y VisitorType
        para conocer el contexto y segmento del visitante.

        **Durante la navegación**

        Utilizar señales como ProductRelated,
        ProductRelated_Duration, BounceRates y ExitRates
        junto con Random Forest para detectar sesiones
        asociadas con mayor intención de compra.

        **Acción de negocio**

        Aplicar intervenciones de bajo costo como
        recomendaciones, mensajes personalizados,
        recordatorios o contenido relevante,
        especialmente durante periodos donde existe
        oportunidad de mejorar la conversión.
        """
    )

    st.subheader(
        "⚠️ Limitaciones"
    )

    st.write(
        """
        - Asociación no significa causalidad.

        - El slider de Mayo es una simulación matemática,
          no una predicción causal.

        - Revenue indica si existió compra o no;
          no representa directamente dinero.

        - El impacto monetario solo puede estimarse
          si se introduce un ticket promedio.

        - VisitorType permite comparar grupos,
          pero no conocer exactamente por qué
          una persona individual no regresó.

        - Feature importance no demuestra que
          una variable cause una compra.
        """
    )

    st.subheader(
        "Conclusión"
    )

    st.success(
        """
        El análisis muestra que el comportamiento de
        navegación contiene información útil para
        diferenciar sesiones que terminan en compra.

        Al combinar:

        **Mayo vs Noviembre**

        **Nuevos vs recurrentes**

        **Exploración interactiva**

        **Simulación del Conversion Rate**

        **Random Forest**

        el dashboard transforma el análisis del Colab
        en una herramienta interactiva que puede apoyar
        decisiones de negocio orientadas a mejorar
        la conversión.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Proyecto Final · Online Shoppers Purchasing Intention ·
    Streamlit · Plotly · Scikit-learn
    """
)
    
    
