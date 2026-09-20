# ============================================================
# PROYECTO FINAL
# ANÁLISIS DEL COMPORTAMIENTO DE LOS USUARIOS EN E-COMMERCE
# ============================================================

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Online Shoppers | Proyecto Final",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background-color: rgba(120,120,120,0.07);
        border: 1px solid rgba(120,120,120,0.16);
        padding: 14px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        font-size: 26px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA Y LIMPIEZA DEL DATASET
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
        csvs = [
            nombre
            for nombre in archivo_zip.namelist()
            if nombre.lower().endswith(".csv")
        ]

        if not csvs:
            raise FileNotFoundError("No se encontró el CSV dentro del ZIP.")

        with archivo_zip.open(csvs[0]) as archivo:
            df_original = pd.read_csv(archivo)

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
    st.write(error)
    st.stop()


# ============================================================
# VARIABLES
# ============================================================

variables_numericas_eda = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# En el Colab se eliminan ProductRelated y Administrative
# después de la matriz de correlación, y PageValues antes del modelo.
variables_eliminadas_modelo = [
    "ProductRelated",
    "Administrative",
    "PageValues",
]

variables_numericas_modelo = [
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

variables_categoricas = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
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
    "Dec",
]

variables_mayo_noviembre = [
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "PageValues",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def tasa_conversion(data):
    if len(data) == 0:
        return 0.0

    return float(
        data["Revenue"].mean() * 100
    )


def con_resultado(data):
    temp = data.copy()

    temp["Resultado"] = temp["Revenue"].map(
        {
            False: "No compra",
            True: "Compra",
        }
    )

    return temp


def safe_percent_change(base, nuevo):
    if base == 0:
        return np.nan

    return (
        (nuevo / base) - 1
    ) * 100


# ============================================================
# MACHINE LEARNING
# ============================================================

@st.cache_resource
def entrenar_modelos(data):

    # Conservamos el dataset completo para EDA,
    # pero replicamos las eliminaciones del Colab SOLO para ML.

    df_modelo = data.copy()

    df_modelo.drop(
        columns=variables_eliminadas_modelo,
        inplace=True,
    )

    X = df_modelo.drop(
        "Revenue",
        axis=1,
    )

    y = df_modelo[
        "Revenue"
    ].astype(int)

    # --------------------------------------------------------
    # ONE HOT ENCODING
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=variables_categoricas,
        dtype=int,
    )

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # ESCALADO
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
    # MODELOS DEL COLAB
    # --------------------------------------------------------

    modelos = {

        "Regresión Logística":
            LogisticRegression(
                solver="saga",
                random_state=42,
            ),

        "Árbol de Decisión":
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=7,
                random_state=42,
            ),

        "k-NN":
            KNeighborsClassifier(
                n_neighbors=25,
                metric="euclidean",
            ),

        "SVM":
            SVC(
                random_state=42,
                kernel="poly",
                C=0.1,
            ),

        "Random Forest":
            RandomForestClassifier(
                class_weight="balanced",
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
            ),
    }

    resultados = []
    matrices = {}

    for nombre, modelo in modelos.items():

        modelo.fit(
            X_train_scaled,
            y_train,
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
                        pred,
                    ),

                "F1 Macro":
                    f1_score(
                        y_test,
                        pred,
                        average="macro",
                        zero_division=0,
                    ),

                "Precision No compra":
                    precision_score(
                        y_test,
                        pred,
                        pos_label=0,
                        zero_division=0,
                    ),

                "Recall No compra":
                    recall_score(
                        y_test,
                        pred,
                        pos_label=0,
                        zero_division=0,
                    ),

                "Precision Compra":
                    precision_score(
                        y_test,
                        pred,
                        pos_label=1,
                        zero_division=0,
                    ),

                "Recall Compra":
                    recall_score(
                        y_test,
                        pred,
                        pos_label=1,
                        zero_division=0,
                    ),
            }
        )

        matrices[nombre] = (
            confusion_matrix(
                y_test,
                pred,
                labels=[0, 1],
            )
        )

    resultados = pd.DataFrame(
        resultados
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    rf = modelos[
        "Random Forest"
    ]

    importancia = (
        pd.DataFrame(
            {
                "Variable":
                    X.columns,

                "Importancia":
                    rf.feature_importances_,
            }
        )
        .sort_values(
            "Importancia",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
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
        "📊 Dataset y EDA",
        "🎛️ Comportamiento vs Revenue",
        "📅 Mayo vs Noviembre",
        "👥 New vs Returning",
        "🛍️ Conversión",
        "🔥 Correlaciones",
        "⚙️ Preparación del modelo",
        "🤖 Modelos",
        "📊 Comparación de modelos",
        "🌲 Random Forest",
        "📈 Variables importantes",
        "💼 Propuesta de negocio",
    ],
)


st.sidebar.divider()

st.sidebar.subheader(
    "Filtros de exploración"
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
        default=meses_disponibles,
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
        default=visitantes_disponibles,
    )
)


dia = st.sidebar.selectbox(
    "Tipo de día",
    [
        "Todos",
        "Entre semana",
        "Fin de semana",
    ],
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


if dia == "Entre semana":

    df_filtrado = df_filtrado[
        df_filtrado[
            "Weekend"
        ] == False
    ]


elif dia == "Fin de semana":

    df_filtrado = df_filtrado[
        df_filtrado[
            "Weekend"
        ] == True
    ]


st.sidebar.metric(
    "Sesiones visibles",
    f"{len(df_filtrado):,}",
)


st.sidebar.caption(
    """
    Los filtros solo afectan el EDA.

    Los modelos siempre usan el dataset limpio completo.
    """
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
        "🛒 Análisis del comportamiento de usuarios en e-commerce"
    )

    st.write(
        """
        El objetivo es identificar qué características del
        comportamiento de navegación se relacionan con la decisión
        de compra y construir un modelo de clasificación que estime
        si una sesión terminará en compra (`Revenue=True`) o no.
        """
    )

    st.info(
        """
        **Hipótesis de negocio:** existen patrones dentro del
        comportamiento de navegación que distinguen sesiones con
        compra de sesiones sin compra y que pueden apoyar decisiones
        dentro de una empresa de e-commerce.
        """
    )

    st.success(
        """
        **Propuesta de negocio:** estudiar qué distingue a noviembre,
        compararlo con mayo —mes con alto volumen de sesiones—,
        segmentar por tipo de visitante y utilizar Random Forest para
        analizar las señales generadas durante la navegación.
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
        f"{total:,}",
    )

    col2.metric(
        "Compras",
        f"{compras:,}",
    )

    col3.metric(
        "No compras",
        f"{total-compras:,}",
    )

    col4.metric(
        "Conversión",
        f"{tasa_conversion(df_filtrado):.2f}%",
    )

    temp = con_resultado(
        df_filtrado
    )

    fig = px.pie(
        temp[
            "Resultado"
        ]
        .value_counts()
        .rename_axis(
            "Resultado"
        )
        .reset_index(
            name="Sesiones"
        ),
        names="Resultado",
        values="Sesiones",
        hole=0.55,
        title="Distribución de Revenue",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.warning(
        """
        Revenue está desbalanceado: la mayoría de las sesiones no
        termina en compra. Por ello, Accuracy no debe interpretarse
        como la única métrica de desempeño.
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
        - Christofer Muñiz Martinez
        """
    )


# ============================================================
# DATASET Y EDA
# ============================================================

elif pagina == "📊 Dataset y EDA":

    st.title(
        "📊 Dataset y EDA"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Registros originales",
        f"{len(df_original):,}",
    )

    col2.metric(
        "Duplicados",
        f"{duplicados:,}",
    )

    col3.metric(
        "Registros limpios",
        f"{len(df):,}",
    )

    col4.metric(
        "Columnas",
        df.shape[1],
    )

    tabs = st.tabs(
        [
            "Datos",
            "Revenue",
            "Variable numérica",
            "VisitorType",
            "Mes",
            "Weekend",
        ]
    )


    # --------------------------------------------------------
    # DATOS
    # --------------------------------------------------------

    with tabs[0]:

        st.dataframe(
            df_filtrado,
            use_container_width=True,
            height=500,
        )


    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    with tabs[1]:

        temp = con_resultado(
            df_filtrado
        )

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

        fig = px.bar(
            conteo,
            x="Resultado",
            y="Sesiones",
            text_auto=True,
            title="Distribución de Revenue",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # VARIABLE NUMÉRICA
    # --------------------------------------------------------

    with tabs[2]:

        variable_eda = (
            st.selectbox(
                "Variable numérica",
                variables_numericas_eda,
                key="variable_eda",
            )
        )

        temp = con_resultado(
            df_filtrado
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            fig = px.histogram(
                temp,
                x=variable_eda,
                color="Resultado",
                nbins=40,
                barmode="overlay",
                title=(
                    f"Distribución de {variable_eda}"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:

            fig = px.box(
                temp,
                x="Resultado",
                y=variable_eda,
                color="Resultado",
                points="outliers",
                title=(
                    f"{variable_eda}: compra vs no compra"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if (
            variable_eda
            == "PageValues"
        ):

            st.info(
                """
                PageValues muestra una asociación fuerte con Revenue,
                pero se elimina antes del modelado por posible fuga
                de objetivo.
                """
            )


    # --------------------------------------------------------
    # VISITOR TYPE
    # --------------------------------------------------------

    with tabs[3]:

        visitante = (
            df_filtrado
            .groupby(
                "VisitorType",
                observed=True,
            )
            .agg(
                Sesiones=(
                    "Revenue",
                    "size",
                ),
                Conversion=(
                    "Revenue",
                    "mean",
                ),
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
            },
            title=(
                "Porcentaje de compra por tipo de visitante"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # MES
    # --------------------------------------------------------

    with tabs[4]:

        resumen_mes = (
            df_filtrado
            .groupby(
                "Month",
                observed=True,
            )
            .agg(
                Sesiones=(
                    "Revenue",
                    "size",
                ),
                Conversion=(
                    "Revenue",
                    "mean",
                ),
            )
            .reset_index()
        )

        resumen_mes[
            "Conversion"
        ] *= 100

        resumen_mes[
            "Month"
        ] = pd.Categorical(
            resumen_mes[
                "Month"
            ],
            categories=orden_meses,
            ordered=True,
        )

        resumen_mes = (
            resumen_mes
            .sort_values(
                "Month"
            )
        )

        fig = px.line(
            resumen_mes,
            x="Month",
            y="Conversion",
            markers=True,
            hover_data=[
                "Sesiones"
            ],
            labels={
                "Conversion":
                    "Conversión (%)",
            },
            title=(
                "Conversión por mes"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # WEEKEND
    # --------------------------------------------------------

    with tabs[5]:

        weekend = (
            df_filtrado
            .groupby(
                "Weekend"
            )
            .agg(
                Sesiones=(
                    "Revenue",
                    "size",
                ),
                Conversion=(
                    "Revenue",
                    "mean",
                ),
            )
            .reset_index()
        )

        weekend[
            "Conversion"
        ] *= 100

        weekend[
            "Tipo"
        ] = weekend[
            "Weekend"
        ].map(
            {
                False:
                    "Entre semana",

                True:
                    "Fin de semana",
            }
        )

        fig = px.bar(
            weekend,
            x="Tipo",
            y="Conversion",
            hover_data=[
                "Sesiones"
            ],
            text_auto=".1f",
            labels={
                "Conversion":
                    "Conversión (%)",
            },
            title=(
                "Conversión: fin de semana vs entre semana"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        "Fuente: UCI Machine Learning Repository — "
        "Online Shoppers Purchasing Intention Dataset."
    )
    # ============================================================
# COMPORTAMIENTO VS REVENUE
# SINGLE-VALUE SLIDERS / UMBRALES
# ============================================================

elif pagina == "🎛️ Comportamiento vs Revenue":

    st.title("🎛️ Comportamiento vs Revenue")

    st.write(
        """
        En lugar de seleccionar rangos, cada slider define **un solo
        valor de referencia**. Las sesiones siguen siendo registros
        reales del dataset; los controles funcionan como umbrales.
        """
    )

    st.success(
        """
        Ejemplo: si bajas el máximo de Exit Rate, el dashboard muestra
        únicamente las sesiones reales que cumplen ese criterio y
        recalcula su tasa de compra.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        min_prod_duration = st.slider(
            "Duración mínima en páginas de producto",
            min_value=0.0,
            max_value=float(
                np.quantile(
                    df_filtrado[
                        "ProductRelated_Duration"
                    ],
                    0.99,
                )
            ),
            value=0.0,
            step=25.0,
        )

        min_admin_duration = st.slider(
            "Duración mínima en páginas administrativas",
            min_value=0.0,
            max_value=float(
                np.quantile(
                    df_filtrado[
                        "Administrative_Duration"
                    ],
                    0.99,
                )
            ),
            value=0.0,
            step=5.0,
        )

        min_informational = st.slider(
            "Mínimo de páginas informativas",
            min_value=int(
                df_filtrado[
                    "Informational"
                ].min()
            ),
            max_value=int(
                df_filtrado[
                    "Informational"
                ].max()
            ),
            value=int(
                df_filtrado[
                    "Informational"
                ].min()
            ),
            step=1,
        )

    with col2:

        max_bounce = st.slider(
            "Bounce Rate máximo",
            min_value=float(
                df_filtrado[
                    "BounceRates"
                ].min()
            ),
            max_value=float(
                df_filtrado[
                    "BounceRates"
                ].max()
            ),
            value=float(
                df_filtrado[
                    "BounceRates"
                ].max()
            ),
            step=0.001,
            format="%.3f",
        )

        max_exit = st.slider(
            "Exit Rate máximo",
            min_value=float(
                df_filtrado[
                    "ExitRates"
                ].min()
            ),
            max_value=float(
                df_filtrado[
                    "ExitRates"
                ].max()
            ),
            value=float(
                df_filtrado[
                    "ExitRates"
                ].max()
            ),
            step=0.001,
            format="%.3f",
        )

        min_info_duration = st.slider(
            "Duración mínima en páginas informativas",
            min_value=0.0,
            max_value=float(
                np.quantile(
                    df_filtrado[
                        "Informational_Duration"
                    ],
                    0.99,
                )
            ),
            value=0.0,
            step=5.0,
        )

    sesiones_seleccionadas = df_filtrado[
        (
            df_filtrado[
                "ProductRelated_Duration"
            ]
            >= min_prod_duration
        )
        &
        (
            df_filtrado[
                "Administrative_Duration"
            ]
            >= min_admin_duration
        )
        &
        (
            df_filtrado[
                "Informational"
            ]
            >= min_informational
        )
        &
        (
            df_filtrado[
                "Informational_Duration"
            ]
            >= min_info_duration
        )
        &
        (
            df_filtrado[
                "BounceRates"
            ]
            <= max_bounce
        )
        &
        (
            df_filtrado[
                "ExitRates"
            ]
            <= max_exit
        )
    ].copy()

    st.divider()

    if len(
        sesiones_seleccionadas
    ) == 0:

        st.warning(
            """
            No existen sesiones reales que cumplan simultáneamente
            con todos los criterios. Ajusta alguno de los sliders.
            """
        )

    else:

        conversion_base = (
            tasa_conversion(
                df_filtrado
            )
        )

        conversion_seleccion = (
            tasa_conversion(
                sesiones_seleccionadas
            )
        )

        compras = int(
            sesiones_seleccionadas[
                "Revenue"
            ].sum()
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Sesiones reales",
            f"{len(sesiones_seleccionadas):,}",
        )

        col2.metric(
            "Compras",
            f"{compras:,}",
        )

        col3.metric(
            "No compras",
            f"{len(sesiones_seleccionadas)-compras:,}",
        )

        col4.metric(
            "Conversión",
            f"{conversion_seleccion:.2f}%",
            delta=(
                f"{conversion_seleccion-conversion_base:+.2f} pp"
            ),
        )

        temp = con_resultado(
            sesiones_seleccionadas
        )

        col1, col2 = st.columns(2)

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
                    "Revenue de las sesiones seleccionadas"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:

            fig = px.scatter(
                temp,
                x="ProductRelated_Duration",
                y="ExitRates",
                color="Resultado",
                opacity=0.60,
                hover_data=[
                    "BounceRates",
                    "Administrative_Duration",
                    "Informational",
                    "Month",
                    "VisitorType",
                ],
                title=(
                    "Duración en producto vs Exit Rate"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.warning(
            """
            Cambiar un umbral no demuestra que modificar esa variable
            cause un aumento en las compras. Esta vista muestra
            asociaciones dentro de las sesiones observadas.
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
        La propuesta de negocio se concentra en entender qué
        diferencias observadas existen entre noviembre y mayo,
        y en cuantificar qué representaría mejorar la conversión
        de mayo.
        """
    )

    df_mayo = df[
        df[
            "Month"
        ] == "May"
    ].copy()

    df_noviembre = df[
        df[
            "Month"
        ] == "Nov"
    ].copy()


    # ========================================================
    # MÉTRICAS GENERALES
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
            f"{sesiones_mayo:,}",
        )

        b.metric(
            "Compras",
            f"{compras_mayo:,}",
        )

        c.metric(
            "Conversión",
            f"{conversion_mayo:.2f}%",
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
            f"{sesiones_noviembre:,}",
        )

        b.metric(
            "Compras",
            f"{compras_noviembre:,}",
        )

        c.metric(
            "Conversión",
            f"{conversion_noviembre:.2f}%",
        )


    comparacion_conversion = (
        pd.DataFrame(
            {
                "Mes": [
                    "Mayo",
                    "Noviembre",
                ],

                "Conversión": [
                    conversion_mayo,
                    conversion_noviembre,
                ],

                "Sesiones": [
                    sesiones_mayo,
                    sesiones_noviembre,
                ],

                "Compras": [
                    compras_mayo,
                    compras_noviembre,
                ],
            }
        )
    )


    fig = px.bar(
        comparacion_conversion,
        x="Mes",
        y="Conversión",
        text_auto=".2f",
        hover_data=[
            "Sesiones",
            "Compras",
        ],
        labels={
            "Conversión":
                "Conversión (%)",
        },
        title=(
            "Conversión: Mayo vs Noviembre"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # QUÉ HACE DIFERENTE A NOVIEMBRE
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 ¿Qué hace diferente a noviembre?"
    )


    variable_comparar = (
        st.selectbox(
            "Variable a comparar",
            variables_mayo_noviembre,
        )
    )


    mayo_promedio = (
        df_mayo[
            variable_comparar
        ].mean()
    )

    noviembre_promedio = (
        df_noviembre[
            variable_comparar
        ].mean()
    )

    diferencia = (
        noviembre_promedio
        - mayo_promedio
    )

    cambio_pct = (
        safe_percent_change(
            mayo_promedio,
            noviembre_promedio,
        )
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Promedio Mayo",
        f"{mayo_promedio:,.3f}",
    )

    col2.metric(
        "Promedio Noviembre",
        f"{noviembre_promedio:,.3f}",
    )

    col3.metric(
        "Diferencia",
        f"{diferencia:+,.3f}",
    )

    col4.metric(
        "Cambio relativo",
        (
            "N/A"
            if pd.isna(
                cambio_pct
            )
            else f"{cambio_pct:+.1f}%"
        ),
    )


    comparacion = pd.concat(
        [
            df_mayo[
                [
                    variable_comparar
                ]
            ].assign(
                Mes="Mayo"
            ),

            df_noviembre[
                [
                    variable_comparar
                ]
            ].assign(
                Mes="Noviembre"
            ),
        ],
        ignore_index=True,
    )


    fig = px.box(
        comparacion,
        x="Mes",
        y=variable_comparar,
        color="Mes",
        points="outliers",
        title=(
            f"{variable_comparar}: "
            "Mayo vs Noviembre"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # TABLA GENERAL
    # ========================================================

    st.subheader(
        "📋 Comparación general"
    )


    resumen = []


    for variable in variables_mayo_noviembre:

        may = (
            df_mayo[
                variable
            ].mean()
        )

        nov = (
            df_noviembre[
                variable
            ].mean()
        )

        resumen.append(
            {
                "Variable":
                    variable,

                "Mayo":
                    may,

                "Noviembre":
                    nov,

                "Diferencia":
                    nov - may,

                "Cambio %":
                    safe_percent_change(
                        may,
                        nov,
                    ),
            }
        )


    st.dataframe(
        pd.DataFrame(
            resumen
        ).round(3),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # MONTH + VISITORTYPE
    # ========================================================

    st.divider()

    st.subheader(
        "👥 Month + VisitorType"
    )

    st.write(
        """
        Igual que en el Colab, eliminamos `Other` en esta parte
        para concentrarnos en New_Visitor y Returning_Visitor.
        """
    )


    df_sin_other = df[
        df[
            "VisitorType"
        ] != "Other"
    ].copy()


    analisis_mes_visitante = (
        df_sin_other
        .groupby(
            [
                "Month",
                "VisitorType",
            ],
            observed=True,
        )
        .agg(
            Total_Sesiones=(
                "Revenue",
                "count",
            ),

            Compras_Exitosas=(
                "Revenue",
                "sum",
            ),

            Tasa_Conversion=(
                "Revenue",
                lambda x:
                    x.mean()
                    * 100,
            ),

            Promedio_Duracion_Producto=(
                "ProductRelated_Duration",
                "mean",
            ),

            Tasa_Salida_Promedio=(
                "ExitRates",
                "mean",
            ),
        )
        .reset_index()
    )


    analisis_mes_visitante[
        "Month"
    ] = pd.Categorical(
        analisis_mes_visitante[
            "Month"
        ],
        categories=orden_meses,
        ordered=True,
    )


    analisis_mes_visitante = (
        analisis_mes_visitante
        .sort_values(
            [
                "Month",
                "VisitorType",
            ]
        )
    )


    tab1, tab2 = st.tabs(
        [
            "Tasa de conversión",
            "Duración en producto",
        ]
    )


    # --------------------------------------------------------
    # CONVERSIÓN POR MES Y TIPO
    # --------------------------------------------------------

    with tab1:

        fig = px.bar(
            analisis_mes_visitante,
            x="Month",
            y="Tasa_Conversion",
            color="VisitorType",
            barmode="group",
            hover_data=[
                "Total_Sesiones",
                "Compras_Exitosas",
            ],
            labels={
                "Tasa_Conversion":
                    "Conversión (%)",

                "VisitorType":
                    "Tipo de visitante",
            },
            title=(
                "Tasa de conversión por mes "
                "y tipo de visitante"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # DURACIÓN EN PRODUCTO
    # --------------------------------------------------------

    with tab2:

        fig = px.bar(
            analisis_mes_visitante,
            x="Month",
            y="Promedio_Duracion_Producto",
            color="VisitorType",
            barmode="group",
            hover_data=[
                "Total_Sesiones",
                "Tasa_Salida_Promedio",
            ],
            labels={
                "Promedio_Duracion_Producto":
                    "Duración promedio",

                "VisitorType":
                    "Tipo de visitante",
            },
            title=(
                "Duración promedio en páginas de producto "
                "por mes y tipo de visitante"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # ========================================================
    # SIMULADOR DE MAYO
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Proyección de mejora del Conversion Rate en Mayo"
    )


    st.info(
        """
        `Revenue` en este dataset es una variable binaria:
        compra / no compra.

        Por eso esta simulación proyecta
        **sesiones con Revenue=True**,
        no dinero directamente.
        """
    )


    max_slider = float(
        max(
            40.0,
            np.ceil(
                conversion_noviembre
                + 10
            ),
        )
    )


    conversion_objetivo = (
        st.slider(
            "Conversion Rate objetivo para Mayo",
            min_value=float(
                round(
                    conversion_mayo,
                    2,
                )
            ),
            max_value=max_slider,
            value=float(
                min(
                    round(
                        max(
                            conversion_mayo,
                            conversion_noviembre,
                        ),
                        2,
                    ),
                    max_slider,
                )
            ),
            step=0.25,
            format="%.2f%%",
        )
    )


    # ========================================================
    # PROYECCIÓN
    # ========================================================

    compras_proyectadas = int(
        round(
            sesiones_mayo
            * conversion_objetivo
            / 100
        )
    )


    compras_adicionales = (
        compras_proyectadas
        - compras_mayo
    )


    mejora_pp = (
        conversion_objetivo
        - conversion_mayo
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Conversión actual",
        f"{conversion_mayo:.2f}%",
    )


    col2.metric(
        "Conversión objetivo",
        f"{conversion_objetivo:.2f}%",
        delta=f"{mejora_pp:+.2f} pp",
    )


    col3.metric(
        "Revenue=True proyectado",
        f"{compras_proyectadas:,}",
    )


    col4.metric(
        "Compras adicionales",
        f"{compras_adicionales:+,}",
    )


    # ========================================================
    # GRÁFICA ACTUAL / OBJETIVO / NOVIEMBRE
    # ========================================================

    escenario = pd.DataFrame(
        {
            "Escenario": [
                "Mayo actual",
                "Mayo proyectado",
                "Noviembre real",
            ],

            "Conversion Rate": [
                conversion_mayo,
                conversion_objetivo,
                conversion_noviembre,
            ],
        }
    )


    fig = px.bar(
        escenario,
        x="Escenario",
        y="Conversion Rate",
        text_auto=".2f",
        labels={
            "Conversion Rate":
                "Conversión (%)",
        },
        title=(
            "Mayo actual vs objetivo vs Noviembre"
        ),
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # CURVA DE PROYECCIÓN
    # ========================================================

    tasas = np.arange(
        float(
            round(
                conversion_mayo,
                2,
            )
        ),
        max_slider + 0.25,
        0.25,
    )


    curva = pd.DataFrame(
        {
            "Conversion Rate":
                tasas,
        }
    )


    curva[
        "Revenue=True proyectado"
    ] = (
        sesiones_mayo
        * curva[
            "Conversion Rate"
        ]
        / 100
    ).round()


    fig = px.line(
        curva,
        x="Conversion Rate",
        y="Revenue=True proyectado",
        labels={
            "Conversion Rate":
                "Conversión (%)",

            "Revenue=True proyectado":
                "Sesiones con compra",
        },
        title=(
            "Compras proyectadas según Conversion Rate"
        ),
    )


    fig.add_vline(
        x=conversion_mayo,
        line_dash="dash",
        annotation_text="Mayo actual",
    )


    fig.add_vline(
        x=conversion_noviembre,
        line_dash="dash",
        annotation_text="Noviembre",
    )


    fig.add_vline(
        x=conversion_objetivo,
        line_dash="dot",
        annotation_text="Objetivo",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.warning(
        """
        La proyección mantiene constante el número de sesiones de
        mayo y cambia únicamente la tasa de conversión.

        No demuestra que replicar una característica de noviembre
        cause por sí sola el resultado proyectado.
        """
    )
    # ============================================================
# NEW VS RETURNING
# ============================================================

elif pagina == "👥 New vs Returning":

    st.title(
        "👥 New Visitor vs Returning Visitor"
    )

    st.write(
        """
        Esta sección compara las sesiones etiquetadas como
        New_Visitor y Returning_Visitor.

        Igual que en el Colab, dejamos fuera la categoría Other
        para concentrarnos en los dos grupos principales.
        """
    )

    df_visitantes = df[
        df[
            "VisitorType"
        ] != "Other"
    ].copy()

    resumen_visitantes = (
        df_visitantes
        .groupby(
            "VisitorType",
            observed=True,
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size",
            ),

            Compras=(
                "Revenue",
                "sum",
            ),

            Conversion=(
                "Revenue",
                "mean",
            ),

            ProductDuration=(
                "ProductRelated_Duration",
                "mean",
            ),

            BounceRate=(
                "BounceRates",
                "mean",
            ),

            ExitRate=(
                "ExitRates",
                "mean",
            ),
        )
        .reset_index()
    )

    resumen_visitantes[
        "Conversion"
    ] *= 100

    resumen_visitantes[
        "% sesiones"
    ] = (
        resumen_visitantes[
            "Sesiones"
        ]
        / resumen_visitantes[
            "Sesiones"
        ].sum()
        * 100
    )

    st.dataframe(
        resumen_visitantes.round(3),
        use_container_width=True,
        hide_index=True,
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
                "Volumen de sesiones por tipo de visitante"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        fig = px.bar(
            resumen_visitantes,
            x="VisitorType",
            y="Conversion",
            text_auto=".1f",
            labels={
                "Conversion":
                    "Conversión (%)",
            },
            title=(
                "Conversión por tipo de visitante"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        """
        Esta comparación no debe interpretarse como una tasa real
        de retención.

        El dataset permite comparar sesiones según VisitorType,
        pero no seguir al mismo usuario individual a lo largo
        del tiempo.
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
        f"{len(df_filtrado):,}",
    )

    col2.metric(
        "Compras",
        f"{compras:,}",
    )

    col3.metric(
        "Conversión",
        f"{conversion:.2f}%",
    )

    # --------------------------------------------------------
    # CONVERSIÓN MENSUAL
    # --------------------------------------------------------

    mensual = (
        df_filtrado
        .groupby(
            "Month",
            observed=True,
        )
        .agg(
            Sesiones=(
                "Revenue",
                "size",
            ),

            Conversion=(
                "Revenue",
                "mean",
            ),
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
        ordered=True,
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
        },
        title=(
            "Conversión por mes"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# CORRELACIONES
# ============================================================

elif pagina == "🔥 Correlaciones":

    st.title(
        "🔥 Matriz de correlación"
    )

    st.write(
        """
        Analizamos las correlaciones entre las variables numéricas
        antes de realizar la selección de variables para el modelo.
        """
    )

    correlacion = (
        df_filtrado[
            variables_numericas_eda
        ]
        .corr()
    )

    fig = px.imshow(
        correlacion,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=(
            "Correlaciones entre variables numéricas"
        ),
    )

    fig.update_layout(
        height=720
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "BounceRates ↔ ExitRates",
        f"""
        {correlacion.loc[
            "BounceRates",
            "ExitRates"
        ]:.2f}
        """,
    )

    col2.metric(
        "ProductRelated ↔ ProductRelated_Duration",
        f"""
        {correlacion.loc[
            "ProductRelated",
            "ProductRelated_Duration"
        ]:.2f}
        """,
    )

    st.write(
        """
        En el Colab se decidió:

        - Mantener **BounceRates** y **ExitRates**.
        - Eliminar **ProductRelated**.
        - Eliminar **Administrative**.
        - Mantener sus variables de duración correspondientes.
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
        "1. Limpieza"
    )

    st.write(
        f"""
        Registros originales:
        **{len(df_original):,}**

        Duplicados eliminados:
        **{duplicados:,}**

        Registros limpios:
        **{len(df):,}**
        """
    )

    st.subheader(
        "2. Variables eliminadas"
    )

    st.write(
        """
        Antes del modelo se eliminan:

        **ProductRelated**

        Se elimina debido a su alta correlación con
        ProductRelated_Duration.

        **Administrative**

        Se elimina para conservar Administrative_Duration
        como representación del comportamiento administrativo.

        **PageValues**

        Se elimina posteriormente por posible fuga de objetivo.
        """
    )

    st.subheader(
        "3. Variables numéricas del modelo"
    )

    st.code(
        """
Administrative_Duration
Informational
Informational_Duration
ProductRelated_Duration
BounceRates
ExitRates
SpecialDay
        """
    )

    st.subheader(
        "4. Variables categóricas"
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

    st.write(
        """
        Las variables categóricas son transformadas mediante
        `pd.get_dummies()`.
        """
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Predictores finales",
        ml[
            "X"
        ].shape[1],
    )

    col2.metric(
        "Train",
        f"{len(ml['X_train']):,}",
    )

    col3.metric(
        "Test",
        f"{len(ml['X_test']):,}",
    )

    st.subheader(
        "5. Train / Test"
    )

    st.write(
        """
        **80% entrenamiento**

        **20% prueba**

        `random_state = 42`

        `stratify = y`
        """
    )

    st.subheader(
        "6. StandardScaler"
    )

    st.write(
        """
        El escalador se ajusta únicamente con los datos
        de entrenamiento y después transforma train y test.
        """
    )

    st.subheader(
        "7. Modelos"
    )

    st.write(
        """
        Se comparan cinco modelos:

        - Regresión Logística
        - Árbol de Decisión
        - k-NN
        - SVM
        - Random Forest
        """
    )


# ============================================================
# MODELOS
# ============================================================

elif pagina == "🤖 Modelos":

    st.title(
        "🤖 Modelos de Machine Learning"
    )

    st.write(
        """
        Selecciona un modelo para revisar sus métricas
        y su matriz de confusión.
        """
    )

    modelo = st.selectbox(
        "Selecciona un modelo",
        ml[
            "resultados"
        ][
            "Modelo"
        ].tolist(),
    )

    fila = (
        ml[
            "resultados"
        ]
        .set_index(
            "Modelo"
        )
        .loc[
            modelo
        ]
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Accuracy",
        f"{fila['Accuracy']*100:.2f}%",
    )

    col2.metric(
        "F1 Macro",
        f"{fila['F1 Macro']:.2f}",
    )

    col3.metric(
        "Precision Compra",
        f"{fila['Precision Compra']*100:.2f}%",
    )

    col4.metric(
        "Recall Compra",
        f"{fila['Recall Compra']*100:.2f}%",
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Precision No compra",
        f"{fila['Precision No compra']*100:.2f}%",
    )

    col2.metric(
        "Recall No compra",
        f"{fila['Recall No compra']*100:.2f}%",
    )

    matriz = (
        ml[
            "matrices"
        ][
            modelo
        ]
    )

    matriz_pct = (
        matriz
        / matriz.sum()
        * 100
    )

    fig = px.imshow(
        matriz_pct,
        text_auto=".1f",
        x=[
            "Predice No compra",
            "Predice Compra",
        ],
        y=[
            "Real No compra",
            "Real Compra",
        ],
        labels={
            "x":
                "Predicción",

            "y":
                "Valor real",

            "color":
                "% del test",
        },
        title=(
            f"Matriz de confusión — {modelo}"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    with st.expander(
        "¿Cómo leer la matriz de confusión?"
    ):

        st.write(
            """
            **Superior izquierda**

            Verdadero negativo:
            No compra clasificada correctamente como No compra.

            **Superior derecha**

            Falso positivo:
            No compra clasificada como Compra.

            **Inferior izquierda**

            Falso negativo:
            Compra clasificada como No compra.

            **Inferior derecha**

            Verdadero positivo:
            Compra clasificada correctamente como Compra.
            """
        )

    if modelo == "SVM":

        st.warning(
            """
            SVM presenta un fuerte sesgo hacia la clase mayoritaria.

            Identifica muy bien No compra, pero prácticamente
            no detecta Compra.

            Por eso una Accuracy alta puede resultar engañosa
            en este problema desbalanceado.
            """
        )

    if modelo == "Random Forest":

        st.info(
            """
            Random Forest detecta una proporción mucho mayor
            de compradores reales, aunque a cambio genera
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

    columnas_pct = [
        "Accuracy",
        "Precision No compra",
        "Recall No compra",
        "Precision Compra",
        "Recall Compra",
    ]

    for columna in columnas_pct:

        tabla[
            columna
        ] = (
            tabla[
                columna
            ]
            * 100
        ).round(2)

    tabla[
        "F1 Macro"
    ] = (
        tabla[
            "F1 Macro"
        ].round(2)
    )

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Accuracy vs F1 Macro"
    )

    comparacion_general = (
        resultados.melt(
            id_vars="Modelo",
            value_vars=[
                "Accuracy",
                "F1 Macro",
            ],
            var_name="Métrica",
            value_name="Valor",
        )
    )

    fig = px.bar(
        comparacion_general,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        text_auto=".2f",
        title=(
            "Comparación general de modelos"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "🎯 Recall para Compra"
    )

    fig = px.bar(
        resultados,
        x="Modelo",
        y="Recall Compra",
        text_auto=".2f",
        title=(
            "Capacidad para detectar compradores reales"
        ),
    )

    fig.update_layout(
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.warning(
        """
        Como Revenue está desbalanceado, Accuracy no debe
        utilizarse como único criterio para elegir el modelo.
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
        f"{rf['Accuracy']*100:.2f}%",
    )

    col2.metric(
        "F1 Macro",
        f"{rf['F1 Macro']:.2f}",
    )

    col3.metric(
        "Recall Compra",
        f"{rf['Recall Compra']*100:.2f}%",
    )

    col4.metric(
        "Precision Compra",
        f"{rf['Precision Compra']*100:.2f}%",
    )

    st.subheader(
        "¿Por qué se selecciona?"
    )

    st.write(
        f"""
        Bajo el criterio de negocio de priorizar la detección
        de compradores potenciales, Random Forest identifica
        aproximadamente:

        **{rf['Recall Compra']*100:.1f}% de las compras reales**
        """
    )

    st.write(
        f"""
        Su Precision para Compra es aproximadamente:

        **{rf['Precision Compra']*100:.1f}%**

        Esto significa que también genera una cantidad
        importante de falsos positivos.
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
            "Predice Compra",
        ],
        y=[
            "Real No compra",
            "Real Compra",
        ],
        labels={
            "x":
                "Predicción",

            "y":
                "Valor real",

            "color":
                "% del test",
        },
        title=(
            "Matriz de confusión — Random Forest"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        """
        Este comportamiento puede ser útil para intervenciones
        de bajo costo como:

        - banners;
        - recomendaciones;
        - recordatorios;
        - personalización de contenido.

        Para incentivos costosos habría que considerar
        cuidadosamente la baja Precision.
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
        El ranking se calcula directamente a partir
        del Random Forest entrenado en este Streamlit.
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
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Top 5 actual"
    )

    for posicion, fila in enumerate(
        importancia.head(5).itertuples(),
        start=1,
    ):

        st.write(
            f"""
            **{posicion}. {fila.Variable}**
            — importancia: {fila.Importancia:.4f}
            """
        )

    st.warning(
        """
        ProductRelated y Administrative no aparecen como
        variables del modelo porque fueron eliminadas antes
        del entrenamiento, tal como se hace en el Colab.

        Sí pueden seguir analizándose dentro del EDA anterior.
        """
    )


# ============================================================
# PROPUESTA DE NEGOCIO
# ============================================================

elif pagina == "💼 Propuesta de negocio":

    st.title(
        "💼 Propuesta de negocio"
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

    mayo = df[
        df[
            "Month"
        ] == "May"
    ]

    noviembre = df[
        df[
            "Month"
        ] == "Nov"
    ]

    conv_mayo = (
        tasa_conversion(
            mayo
        )
    )

    conv_nov = (
        tasa_conversion(
            noviembre
        )
    )

    st.subheader(
        "1. Problema"
    )

    st.write(
        """
        El volumen de sesiones no está directamente relacionado
        con una mayor conversión.

        Por ello, nos interesa estudiar por qué un mes como
        noviembre presenta un comportamiento diferente a mayo.
        """
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Conversión Mayo",
        f"{conv_mayo:.2f}%",
    )

    col2.metric(
        "Conversión Noviembre",
        f"{conv_nov:.2f}%",
    )

    st.subheader(
        "2. Qué hace diferente a Noviembre"
    )

    st.write(
        """
        La propuesta es estudiar especialmente:

        - ExitRates
        - BounceRates
        - ProductRelated_Duration
        - Administrative_Duration
        - Month
        - VisitorType

        para identificar patrones asociados con la mayor
        conversión observada.
        """
    )

    st.subheader(
        "3. Month + VisitorType"
    )

    st.write(
        """
        Month y VisitorType pueden utilizarse como una
        segmentación descriptiva inicial.

        Esto permite entender quién llega y en qué contexto
        temporal antes de analizar su comportamiento dentro
        de la página.
        """
    )

    st.subheader(
        "4. Random Forest durante la navegación"
    )

    st.write(
        f"""
        Random Forest permite detectar aproximadamente:

        **{rf['Recall Compra']*100:.1f}% de las sesiones que
        realmente terminan en compra.**

        Su Precision para Compra es aproximadamente:

        **{rf['Precision Compra']*100:.1f}%**
        """
    )

    st.subheader(
        "5. Aplicar los hallazgos a Mayo"
    )

    st.write(
        """
        El objetivo no es afirmar que noviembre pueda copiarse
        directamente.

        La idea es identificar qué características de navegación
        son diferentes y utilizarlas para diseñar hipótesis de
        mejora durante mayo.
        """
    )

    st.info(
        """
        En la sección **Mayo vs Noviembre**, el slider permite
        cuantificar cuántas sesiones con Revenue=True
        corresponderían a distintos objetivos de conversión
        manteniendo constante el volumen de mayo.
        """
    )

    st.subheader(
        "6. Acciones posibles"
    )

    st.success(
        """
        Algunas acciones de bajo costo que podrían probarse son:

        - recomendaciones de producto;
        - contenido más relevante;
        - mejoras de UX;
        - recordatorios;
        - personalización;
        - mensajes antes del abandono.

        El impacto real de estas acciones tendría que evaluarse
        mediante experimentos controlados.
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

        - Revenue indica compra/no compra, no ingreso monetario.

        - VisitorType no permite medir retención individual.

        - Random Forest presenta falsos positivos.

        - ProductRelated y Administrative fueron eliminadas
          antes del modelado, aunque siguen disponibles en el EDA.
        """
    )

    st.subheader(
        "Conclusión"
    )

    st.success(
        """
        El análisis muestra que existen patrones dentro del
        comportamiento de navegación asociados con Revenue.

        Al combinar el EDA, la comparación Mayo vs Noviembre,
        VisitorType, el simulador de conversión y Random Forest,
        el dashboard convierte los resultados del análisis
        en una herramienta interactiva para apoyar decisiones
        de negocio.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Proyecto Final ·
    Online Shoppers Purchasing Intention ·
    Streamlit · Plotly · Scikit-learn
    """
)
    
