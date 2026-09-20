import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
# CONFIGURACIÓN
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

        .info-box {
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(120,120,120,0.20);
            background-color: rgba(120,120,120,0.06);
            margin-bottom: 15px;
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

    with zipfile.ZipFile(io.BytesIO(contenido)) as archivo_zip:

        archivos_csv = [
            nombre
            for nombre in archivo_zip.namelist()
            if nombre.lower().endswith(".csv")
        ]

        if not archivos_csv:
            raise FileNotFoundError(
                "No se encontró el CSV dentro del archivo."
            )

        with archivo_zip.open(archivos_csv[0]) as archivo:
            df_original = pd.read_csv(archivo)

    # ========================================================
    # LIMPIEZA REALIZADA EN EL COLAB
    # ========================================================

    duplicados = int(
        df_original.duplicated().sum()
    )

    df_limpio = (
        df_original
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_original, df_limpio, duplicados


try:
    df_original, df, duplicados = cargar_datos()

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
        data["Revenue"].mean() * 100
    )


def agregar_resultado(data):

    temp = data.copy()

    temp["Resultado"] = (
        temp["Revenue"]
        .map({
            False: "No compra",
            True: "Compra"
        })
    )

    return temp


def paso_slider(serie):

    rango = (
        float(serie.max())
        - float(serie.min())
    )

    if rango <= 1:
        return 0.001

    if rango <= 10:
        return 0.1

    if rango <= 100:
        return 1.0

    return max(
        1.0,
        round(rango / 500, 2)
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

@st.cache_resource
def entrenar_modelos(data):

    # ========================================================
    # COPIA DEL DATASET LIMPIO
    # ========================================================

    df_modelo = data.copy()

    # ========================================================
    # PAGEVALUES SE EXCLUYE DEL MODELO
    # ========================================================

    df_modelo.drop(
        columns=["PageValues"],
        inplace=True
    )

    # ========================================================
    # X / Y
    # ========================================================

    X = df_modelo.drop(
        "Revenue",
        axis=1
    )

    y = (
        df_modelo["Revenue"]
        .astype(int)
    )

    # ========================================================
    # DUMMIES
    # ========================================================

    X = pd.get_dummies(
        X,
        columns=variables_categoricas,
        dtype=int
    )

    # ========================================================
    # TRAIN / TEST
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # ========================================================
    # ESCALADO
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

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

    # ========================================================
    # LOS 5 MODELOS DEL COLAB
    # ========================================================

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

    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    for nombre, modelo in modelos.items():

        modelo.fit(
            X_train_scaled,
            y_train
        )

        pred = modelo.predict(
            X_test_scaled
        )

        resultados.append({

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
        })

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

    # ========================================================
    # IMPORTANCIA RANDOM FOREST
    # ========================================================

    rf = modelos[
        "Random Forest"
    ]

    importancia = pd.DataFrame({
        "Variable":
            X.columns,

        "Importancia":
            rf.feature_importances_
    })

    importancia = (
        importancia
        .sort_values(
            "Importancia",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return {

        "X": X,

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
    "Preparando modelos de Machine Learning..."
):
    ml = entrenar_modelos(df)


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
        "🛍️ Conversión",
        "🔥 Relaciones entre variables",
        "⚙️ Preparación del modelo",
        "🤖 Modelos de Machine Learning",
        "📊 Comparación de modelos",
        "🌲 Random Forest",
        "📈 Variables importantes",
        "💡 Propuesta y conclusiones"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "Filtros generales"
)

meses_disponibles = [
    mes
    for mes in orden_meses
    if mes in df["Month"].unique()
]

meses_seleccionados = (
    st.sidebar.multiselect(
        "Mes",
        meses_disponibles,
        default=meses_disponibles
    )
)

visitantes_disponibles = sorted(
    df["VisitorType"]
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
# FILTRO GENERAL SOBRE EL DATASET LIMPIO
# ============================================================

df_filtrado = df[
    df["Month"].isin(
        meses_seleccionados
    )
    &
    df["VisitorType"].isin(
        visitantes_seleccionados
    )
].copy()

if dia_seleccionado == "Entre semana":

    df_filtrado = df_filtrado[
        df_filtrado["Weekend"] == False
    ]

elif dia_seleccionado == "Fin de semana":

    df_filtrado = df_filtrado[
        df_filtrado["Weekend"] == True
    ]


st.sidebar.metric(
    "Sesiones visibles",
    f"{len(df_filtrado):,}"
)

st.sidebar.caption(
    "Los filtros trabajan sobre el dataset limpio. "
    "El entrenamiento de los modelos siempre utiliza "
    "todo el dataset limpio."
)


if len(df_filtrado) == 0:

    st.warning(
        "No existen sesiones para los filtros seleccionados."
    )

    st.stop()


# ============================================================
# INICIO
# ============================================================

if pagina == "🏠 Inicio":

    st.title(
        "🛒 Análisis del comportamiento de los usuarios "
        "durante el proceso de compra"
    )

    st.subheader(
        "Online Shoppers Purchasing Intention"
    )

    st.write(
        """
        Este dashboard transforma el análisis realizado
        en Google Colab en una herramienta interactiva para
        explorar el comportamiento de los usuarios dentro
        de una plataforma de e-commerce.
        """
    )

    st.info(
        """
        **Objetivo**

        Analizar qué características del comportamiento de
        navegación están relacionadas con la decisión de
        compra y desarrollar modelos de clasificación para
        distinguir sesiones que terminan en compra de aquellas
        que no.
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

    col1, col2 = st.columns(
        [1.15, 1]
    )

    with col1:

        temp = agregar_resultado(
            df_filtrado
        )

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
            title=(
                "Distribución de la variable objetivo Revenue"
            )
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
            **True:** la sesión terminó en compra.

            **False:** la sesión no terminó en compra.
            """
        )

        st.warning(
            """
            Revenue se encuentra desbalanceada:
            la mayor parte de las sesiones no terminan
            en compra. Por ello, Accuracy no debe
            interpretarse de manera aislada.
            """
        )

    st.divider()

    st.subheader(
        "¿Qué hace interactivo este dashboard?"
    )

    st.write(
        """
        Puedes modificar rangos de comportamiento como la
        cantidad de páginas de producto visitadas, el tiempo
        dedicado a productos, Bounce Rate y Exit Rate.

        Al modificar los controles, el dashboard filtra
        sesiones que realmente existen en el dataset limpio
        y vuelve a calcular Revenue y la tasa de compra.
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
        Los datos mostrados aquí corresponden al mismo
        procesamiento utilizado en el Colab del proyecto.
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
        "Duplicados encontrados",
        f"{duplicados:,}"
    )

    col3.metric(
        "Filas después de limpiar",
        f"{len(df):,}"
    )

    col4.metric(
        "Columnas",
        df.shape[1]
    )

    st.success(
        f"""
        Se eliminaron {duplicados} registros duplicados.
        El análisis interactivo utiliza {len(df):,}
        registros del dataset limpio.
        """
    )

    tab1, tab2, tab3, tab4 = (
        st.tabs(
            [
                "Vista de datos",
                "Estadísticas",
                "Calidad",
                "Variables"
            ]
        )
    )

    with tab1:

        st.dataframe(
            df_filtrado,
            use_container_width=True,
            height=520
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

        st.write(
            """
            El dataset no presenta valores faltantes.
            Los duplicados fueron eliminados antes de
            realizar la exploración y el modelado.
            """
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


# ============================================================
# EXPLORACIÓN INTERACTIVA
# ============================================================

elif pagina == "🔎 Exploración interactiva":

    st.title(
        "🔎 Exploración interactiva"
    )

    st.write(
        """
        Selecciona una variable del análisis exploratorio
        y modifica su rango. Todas las observaciones
        provienen del dataset limpio.
        """
    )

    variable = st.selectbox(
        "Variable a analizar",
        variables_numericas
    )

    minimo = float(
        df_filtrado[
            variable
        ].min()
    )

    maximo = float(
        df_filtrado[
            variable
        ].max()
    )

    if maximo > minimo:

        rango = st.slider(
            f"Rango de {variable}",
            min_value=minimo,
            max_value=maximo,
            value=(
                minimo,
                maximo
            ),
            step=paso_slider(
                df_filtrado[
                    variable
                ]
            )
        )

        df_exploracion = (
            df_filtrado[
                (
                    df_filtrado[
                        variable
                    ] >= rango[0]
                )
                &
                (
                    df_filtrado[
                        variable
                    ] <= rango[1]
                )
            ].copy()
        )

    else:

        df_exploracion = (
            df_filtrado.copy()
        )

    if len(
        df_exploracion
    ) == 0:

        st.warning(
            "No existen sesiones dentro del rango seleccionado."
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
        delta=(
            f"{diferencia:+.2f} pp"
        )
    )

    st.caption(
        "El delta compara el rango seleccionado "
        "contra la tasa de conversión de los datos "
        "actualmente visibles."
    )

    col1, col2 = (
        st.columns(2)
    )

    temp = agregar_resultado(
        df_exploracion
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
        "📈 Cómo cambia Revenue"
    )

    datos_rangos = (
        df_filtrado[
            [variable, "Revenue"]
        ]
        .dropna()
        .copy()
    )

    try:

        datos_rangos[
            "Rango"
        ] = pd.qcut(
            datos_rangos[
                variable
            ],
            q=8,
            duplicates="drop"
        )

    except ValueError:

        datos_rangos[
            "Rango"
        ] = pd.cut(
            datos_rangos[
                variable
            ],
            bins=8,
            duplicates="drop"
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
            **ProductRelated** representa la cantidad de
            páginas relacionadas con productos visitadas
            durante una sesión.

            Dentro de este dataset es la variable más cercana
            a la idea de analizar las interacciones o
            "clicks" relacionados con productos antes de
            finalizar una compra.
            """
        )

    if variable == "PageValues":

        st.warning(
            """
            PageValues se conserva en el análisis exploratorio
            porque fue una variable importante durante el EDA.

            Sin embargo, no se utiliza como predictor en el
            modelo de Machine Learning debido al riesgo de
            target leakage.
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
        Esta sección permite modificar simultáneamente
        características relacionadas con el comportamiento
        de navegación y observar cómo cambia Revenue.
        """
    )

    st.success(
        """
        Los sliders NO generan usuarios artificiales.
        Cada movimiento filtra sesiones que realmente
        existen dentro del dataset limpio del proyecto.
        """
    )

    st.caption(
        "ProductRelated se utiliza como aproximación a "
        "la cantidad de interacciones con páginas de producto."
    )

    # ========================================================
    # RANGOS
    # ========================================================

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

    # ========================================================
    # FILTRADO DE SESIONES REALES
    # ========================================================

    simulacion = df_filtrado[
        (
            df_filtrado[
                "ProductRelated"
            ].between(
                rango_prod[0],
                rango_prod[1]
            )
        )
        &
        (
            df_filtrado[
                "ProductRelated_Duration"
            ].between(
                rango_prod_dur[0],
                rango_prod_dur[1]
            )
        )
        &
        (
            df_filtrado[
                "BounceRates"
            ].between(
                rango_bounce[0],
                rango_bounce[1]
            )
        )
        &
        (
            df_filtrado[
                "ExitRates"
            ].between(
                rango_exit[0],
                rango_exit[1]
            )
        )
        &
        (
            df_filtrado[
                "Administrative"
            ].between(
                rango_admin[0],
                rango_admin[1]
            )
        )
        &
        (
            df_filtrado[
                "Informational"
            ].between(
                rango_info[0],
                rango_info[1]
            )
        )
    ].copy()

    st.divider()

    if len(simulacion) == 0:

        st.warning(
            """
            No existen sesiones reales que cumplan
            simultáneamente con todos los parámetros
            seleccionados.

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
            "Sesiones reales encontradas",
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
            delta=(
                f"{cambio:+.2f} pp"
            )
        )

        st.caption(
            f"Tasa de compra del conjunto de referencia: "
            f"{conversion_original:.2f}%."
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
                hole=0.50,
                title=(
                    "Revenue en las sesiones seleccionadas"
                )
            )

            fig.update_traces(
                textinfo="percent+label"
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
                    "Páginas de producto visitadas"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "Interacción con productos"
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
                "Cantidad de páginas vs tiempo "
                "en páginas de producto"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Comparación de comportamiento"
        )

        resumen = pd.DataFrame(
            {
                "Métrica": [
                    "ProductRelated",
                    "ProductRelated_Duration",
                    "BounceRates",
                    "ExitRates",
                    "Administrative",
                    "Informational"
                ],

                "Dataset visible": [
                    df_filtrado[
                        "ProductRelated"
                    ].mean(),

                    df_filtrado[
                        "ProductRelated_Duration"
                    ].mean(),

                    df_filtrado[
                        "BounceRates"
                    ].mean(),

                    df_filtrado[
                        "ExitRates"
                    ].mean(),

                    df_filtrado[
                        "Administrative"
                    ].mean(),

                    df_filtrado[
                        "Informational"
                    ].mean()
                ],

                "Sesiones seleccionadas": [
                    simulacion[
                        "ProductRelated"
                    ].mean(),

                    simulacion[
                        "ProductRelated_Duration"
                    ].mean(),

                    simulacion[
                        "BounceRates"
                    ].mean(),

                    simulacion[
                        "ExitRates"
                    ].mean(),

                    simulacion[
                        "Administrative"
                    ].mean(),

                    simulacion[
                        "Informational"
                    ].mean()
                ]
            }
        )

        resumen_long = resumen.melt(
            id_vars="Métrica",
            var_name="Grupo",
            value_name="Valor"
        )

        fig = px.bar(
            resumen_long,
            x="Métrica",
            y="Valor",
            color="Grupo",
            barmode="group",
            title=(
                "Promedios del comportamiento seleccionado"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.warning(
            """
            Estos resultados muestran asociaciones presentes
            en el dataset. No demuestran que modificar una
            variable cause directamente un cambio en la compra.
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

    # ========================================================
    # VISITOR TYPE
    # ========================================================

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
            "Conversion":
                "Conversión (%)",
            "VisitorType":
                "Tipo de visitante"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # MONTH
    # ========================================================

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
            "Conversion":
                "Conversión (%)",
            "Month":
                "Mes"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # WEEKEND
    # ========================================================

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
        }
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
        Selecciona dos variables para estudiar cómo se
        relacionan entre sí y cómo se distribuyen las
        sesiones con y sin compra.
        """
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        variable_x = st.selectbox(
            "Variable X",
            variables_numericas,
            index=4
        )

    with col2:

        variable_y = st.selectbox(
            "Variable Y",
            variables_numericas,
            index=5
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
        "Correlación entre las variables",
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
            "VisitorType",
            "Revenue"
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

    matriz_correlacion = (
        df_filtrado[
            variables_numericas
        ]
        .corr()
    )

    fig = px.imshow(
        matriz_correlacion,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=(
            "Correlaciones de las variables numéricas"
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
        {matriz_correlacion.loc[
            "BounceRates",
            "ExitRates"
        ]:.2f}
        """
    )

    col2.metric(
        "ProductRelated ↔ Duration",
        f"""
        {matriz_correlacion.loc[
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
        El dataset original contiene
        **{len(df_original):,} registros**.

        Se detectaron y eliminaron
        **{duplicados} registros duplicados**.

        El dataset limpio contiene
        **{len(df):,} sesiones**.
        """
    )

    st.subheader(
        "2. Eliminación de PageValues para ML"
    )

    st.warning(
        """
        PageValues permanece disponible para el análisis
        exploratorio, pero se elimina de los predictores
        antes del entrenamiento debido al posible riesgo
        de target leakage.
        """
    )

    st.subheader(
        "3. Codificación de variables categóricas"
    )

    st.write(
        """
        Se aplica One-Hot Encoding mediante `pd.get_dummies`
        a Month, OperatingSystems, Browser, Region,
        TrafficType, VisitorType y Weekend.
        """
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Predictores después de dummies",
        ml["X"].shape[1]
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
        Se utiliza 80% para entrenamiento y 20% para prueba,
        con `random_state=42` y `stratify=y` para conservar
        aproximadamente la proporción de Revenue.
        """
    )

    st.subheader(
        "5. StandardScaler"
    )

    st.write(
        """
        Las variables numéricas se estandarizan utilizando
        StandardScaler ajustado únicamente con el conjunto
        de entrenamiento.
        """
    )


# ============================================================
# MODELOS
# ============================================================

elif pagina == "🤖 Modelos de Machine Learning":

    st.title(
        "🤖 Modelos de Machine Learning"
    )

    st.write(
        """
        En el Colab se compararon cinco algoritmos de
        clasificación utilizando las mismas particiones
        de entrenamiento y prueba.
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

    matriz = (
        ml[
            "matrices"
        ][
            modelo_seleccionado
        ]
    )

    matriz_pct = (
        matriz
        /
        matriz.sum()
        *
        100
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
                "% del conjunto test"
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

    if modelo_seleccionado == "SVM":

        st.warning(
            """
            En los resultados del proyecto, SVM presenta
            un fuerte sesgo hacia la clase mayoritaria.
            Una Accuracy alta no implica necesariamente
            una buena capacidad para detectar compradores.
            """
        )

    if modelo_seleccionado == "Random Forest":

        st.info(
            """
            Random Forest detecta una proporción mucho mayor
            de las sesiones que realmente terminan en compra,
            aunque a costa de generar más falsos positivos.
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

        tabla[columna] = (
            tabla[columna]
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
        resultados
        .melt(
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
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Detección de compradores"
    )

    fig = px.bar(
        resultados,
        x="Modelo",
        y="Recall Compra",
        text_auto=".2f",
        labels={
            "Recall Compra":
                "Recall Compra"
        },
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
        Debido al desbalance de Revenue, una Accuracy
        elevada puede ser engañosa si el modelo clasifica
        casi todas las sesiones como No compra.
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
        "Criterio utilizado en el proyecto"
    )

    st.write(
        f"""
        Bajo el criterio de negocio de priorizar la
        identificación de compradores potenciales,
        Random Forest logra detectar aproximadamente
        **{rf['Recall Compra']*100:.1f}% de las compras
        reales del conjunto de prueba**.
        """
    )

    st.write(
        f"""
        Sin embargo, su Precision para la clase Compra es
        aproximadamente **{rf['Precision Compra']*100:.1f}%**,
        por lo que también genera una cantidad importante
        de falsos positivos.
        """
    )

    matriz = (
        ml[
            "matrices"
        ][
            "Random Forest"
        ]
    )

    matriz_pct = (
        matriz
        /
        matriz.sum()
        *
        100
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
        Este comportamiento puede ser útil para
        intervenciones de bajo costo, como recomendaciones,
        recordatorios o mensajes personalizados.

        Para promociones costosas, la baja Precision debe
        considerarse antes de actuar sobre todas las sesiones
        identificadas por el modelo.
        """
    )


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

elif pagina == "📈 Variables importantes":

    st.title(
        "📈 Variables importantes"
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
        "Top 5"
    )

    top5 = (
        importancia
        .head(5)
    )

    for posicion, fila in enumerate(
        top5.itertuples(),
        start=1
    ):

        st.write(
            f"**{posicion}. {fila.Variable}** "
            f"— importancia: "
            f"{fila.Importancia:.4f}"
        )

    st.info(
        """
        En los resultados del proyecto destacan variables
        relacionadas con ExitRates, tiempo de interacción
        con productos, BounceRates y cantidad de páginas
        visitadas.

        La importancia de una variable no demuestra
        causalidad.
        """
    )


# ============================================================
# PROPUESTA Y CONCLUSIONES
# ============================================================

elif pagina == "💡 Propuesta y conclusiones":

    st.title(
        "💡 Propuesta de negocio y conclusiones"
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

    st.subheader(
        "1. El comportamiento de navegación contiene información útil"
    )

    st.write(
        """
        Las sesiones que terminan en compra presentan
        diferencias observables en variables relacionadas
        con interacción con productos, permanencia,
        Bounce Rate y Exit Rate.
        """
    )

    st.subheader(
        "2. Interacción con productos"
    )

    st.write(
        """
        ProductRelated y ProductRelated_Duration permiten
        estudiar qué ocurre cuando los usuarios visitan
        más páginas de producto o permanecen más tiempo
        interactuando con ellas.
        """
    )

    st.subheader(
        "3. Abandono del sitio"
    )

    st.write(
        """
        BounceRates y ExitRates presentan una relación
        importante con el comportamiento de compra.
        Además, ambas variables muestran una correlación
        elevada entre sí.
        """
    )

    st.subheader(
        "4. Propuesta de negocio"
    )

    st.write(
        """
        La empresa podría utilizar estas señales de
        comportamiento para identificar sesiones con
        características asociadas a una mayor intención
        de compra y aplicar intervenciones de bajo costo,
        como recomendaciones, mensajes personalizados o
        mejoras en la experiencia de navegación.
        """
    )

    st.subheader(
        "5. Machine Learning"
    )

    st.write(
        f"""
        Bajo el criterio del proyecto de priorizar la
        detección de compradores, Random Forest alcanza
        aproximadamente **{rf['Recall Compra']*100:.1f}%**
        de Recall para la clase Compra.
        """
    )

    st.warning(
        f"""
        La Precision para Compra es aproximadamente
        **{rf['Precision Compra']*100:.1f}%**.

        Por ello, el modelo no debería utilizarse
        automáticamente para entregar incentivos costosos
        a cada sesión identificada como comprador potencial.
        """
    )

    st.subheader(
        "Conclusión"
    )

    st.success(
        """
        El análisis exploratorio y los modelos respaldan
        la hipótesis de que el comportamiento de navegación
        contiene información relevante para distinguir
        sesiones que terminan en compra de aquellas que no.

        El dashboard permite convertir los hallazgos del
        Colab en una herramienta interactiva donde pueden
        modificarse parámetros y observar cómo cambia
        Revenue utilizando únicamente sesiones reales del
        dataset limpio.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Proyecto Final · Online Shoppers Purchasing Intention · "
    "Streamlit · Plotly · Scikit-learn"
)
