#Creamos el archivo de la APP en el interprete principal (Python)

####################IMPLEMENTACION DE DASHBOARD####################

#Verificamos que todas las librerias se puedan importar
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from funpymodeling.exploratory import freq_tbl
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import plotly.express as px
import base64

###################################################################
#Configuración de la página de Streamlit
st.set_page_config(page_title="Dashboard Tokyo", page_icon=":bar_chart:", layout="wide")
#Definimos la instancia de streamlit
@st.cache_resource
#Creamos la función de carga de datos
def load_data():
    #Lectura del archivo csv con índice
    df = pd.read_csv("tokyoC.csv")

    #Selección de columnas numéricas del dataframe principal
    numeric_df = df.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns

    #Función para análisis univariado y filtrado
    def univariate_analysis(variable):
        table = freq_tbl(df[variable])  #Genera tabla de frecuencias
        filtro = table[table['frequency'] > 1]  #Filtro para valores con frecuencia mayor a 1
        filtro_index = filtro.set_index(variable)  #Ajusta el índice al valor de la variable
        numeric_df = filtro_index.select_dtypes(['float', 'int'])  #Selección de columnas numéricas
        return filtro_index, numeric_df.columns

    #Realizamos el análisis univariado para las variables específicas
    Filtro_index1, numeric_cols1 = univariate_analysis('property_type')
    Filtro_index2, numeric_cols2 = univariate_analysis('host_name')
    Filtro_index3, numeric_cols3 = univariate_analysis('host_location')
    Filtro_index4, numeric_cols4 = univariate_analysis('room_type')
    Filtro_index5, numeric_cols5 = univariate_analysis('host_response_time')

    #Devolvemos los dataframes y las columnas numéricas resultantes
    return Filtro_index1, Filtro_index2, Filtro_index3, Filtro_index4, Filtro_index5, df, numeric_df, numeric_cols, numeric_cols2, numeric_cols3, numeric_cols4, numeric_cols5
    

#Cargar los datos obtenidos
Filtro_index1, Filtro_index2, Filtro_index3, Filtro_index4, Filtro_index5, df, numeric_df, numeric_cols, numeric_cols2, numeric_cols3, numeric_cols4, numeric_cols5 = load_data()

#SIDEBAR
st.sidebar.title(":jp: Dashboard Tokyo :jp:")
st.sidebar.header("Panel de opciones")

#Insertar la imagen en sidebar
#URL of the image
image = "https://www.vice.com/wp-content/uploads/sites/2/2024/07/1408551938bc55r.gif"  
st.sidebar.image(image, use_column_width=True)


#FRAMES
Frames = st.selectbox(label="Análisis de correlaciones", options=['Portada',"Regresion lineal simple", "Regresion lineal multiple", "Regresion no lineal", "Boxplot"])

if Frames == "Portada":
    #Mostrar imágenes o gifs
    st.markdown("![TOKYO](https://i.pinimg.com/originals/2a/7f/79/2a7f796fb55b5557b381d800af60735d.gif)")
    button=st.sidebar.button(label="APRIÉTAME")
    if button:
        st.markdown(""" 
            <div style='text-align:justified;'>
                <p style='font-size:20px;color:white;'>
                    Tokio, la ajetreada capital de Japón, mezcla lo ultramoderno y lo tradicional, desde los rascacielos iluminados con neones hasta los templos históricos. El opulento santuario Shinto Meiji es conocido por su puerta altísima y los bosques circundantes. El Palacio Imperial se ubica en medio de grandes jardines públicos. Los distintos museos de la ciudad ofrecen exhibiciones que van desde el arte clásico (en el Museo Nacional de Tokio) hasta un teatro kabuki reconstruido (en el Museo Edo-Tokyo).
            """,unsafe_allow_html=True)
        image_list = [
        ("Shinto_meiji.jpg", "Santuario Shino Meiji"),
        ("Castilloimperial.jpg", "Palacio Imperial"),
        ("museo_nacional.jpg", "Museo Nacional de Japón")]


    # Crear columnas para mostrar las imágenes en una distribución de 3 columnas
        cols = st.columns(3)  # Crea 3 columnas para las imágenes
        for i, (img_path, caption) in enumerate(image_list):
            with cols[i]:
                st.image(img_path, caption=f"{caption}", use_column_width=True)    
##############################################################################################################################

#Regresion lineal simple
if Frames == "Regresion lineal simple":
    #Coloreamos el título
    st.markdown("<h1 style='text-align: center; color: slateblue;'> Regresión lineal simple</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: aliceblue;'>Selecciona las variables desde la Sidebar</h2>", unsafe_allow_html=True)

    #Personalizamos la Sidebar
    st.sidebar.markdown("<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.sidebar.markdown("**Opciones de visualización**", unsafe_allow_html=True)

    check_box = st.sidebar.checkbox(label="Mostrar Dataset")
    if check_box:
        st.write(Filtro_index1)
        st.write(df)

    x_selected = st.sidebar.selectbox(label="Escoge tu x", options=numeric_cols)
    y_selected = st.sidebar.selectbox(label="Escoge tu y", options=numeric_cols)

    #Modelo y resultados
    x = numeric_df[x_selected].values.reshape(-1, 1)
    y = numeric_df[y_selected].values
    model = LinearRegression()
    model.fit(x, y)
    coef_Deter = model.score(x, y)
    coef_Correl = np.sqrt(coef_Deter)

    #Predecir los valores de y para la línea de tendencia
    y_pred = model.predict(x)

    #Crear gráfico de dispersión con la línea de tendencia
    figure1 = px.scatter(data_frame=numeric_df, x=x_selected, y=y_selected,
                         title='Regresión lineal simple', color_discrete_sequence=["#db86cd"],
                         template="plotly_dark")
    
    #Agregar línea de tendencia
    figure1.add_scatter(x=numeric_df[x_selected], y=y_pred, mode='lines', name='Línea de tendencia', line=dict(color='red', width=2))

    #Mostrar gráfico
    st.plotly_chart(figure1)

    st.markdown(f"<p style='font-size: 16px; color: white;'> <strong>Coeficiente de correlación (R):</strong> {coef_Correl:.4f}</p>", unsafe_allow_html=True)

#############################################################################################################

#Regresión lineal múltiple
if Frames == "Regresion lineal multiple":
    st.markdown("<h1 style='text-align: center; color: slateblue;'> Regresión lineal múltiple</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: aliceblue;'>Selecciona las variables desde la sidebar</h2>", unsafe_allow_html=True)

    #Seleccionar la variable dependiente (y) y las independientes (x)
    var_dep = st.sidebar.selectbox(label="Variable dependiente", options=numeric_cols) 
    vars_indep = st.sidebar.multiselect(label="Variables independientes", options=numeric_cols)

    if vars_indep and var_dep:
        #Declaramos las variables dependientes e independientes
        Vars_Indep = df[vars_indep]
        Var_Dep = df[var_dep]

        #Definimos y ajustamos el modelo de regresión lineal múltple
        model = LinearRegression()
        model.fit(X=Vars_Indep, y=Var_Dep)

        #Predecimos los valores usando el modelo ajustado 
        y_pred = model.predict(X=Vars_Indep)

        #Insertamos las predicciones en el DataFrame
        df['Predicciones'] = y_pred

        #Visualizamos los gráficos 
        for var in vars_indep:
            #Gráfico de dispersión de la variable independiente vs total real
            figure = px.scatter(df, x=var, y=var_dep, title=f'Relación entre {var} y {var_dep}',
                                labels={var: var, var_dep: var_dep},
                                color_discrete_sequence=["pink"], opacity=0.5)

            #Agregar las predicciones como puntos
            figure.add_scatter(x=df[var], y=df['Predicciones'], mode='markers', name='Predicciones', 
                               marker=dict(color='red', size=8, symbol='diamond', line=dict(width=2, color='black')), opacity=0.5)

            #Mostrar el gráfico
            st.plotly_chart(figure)

        #Coeficiente de determinación (R²) y coeficiente de correlación
        coef_Deter = model.score(Vars_Indep, Var_Dep)
        coef_Correl = np.sqrt(coef_Deter)

        st.markdown(f"<p style='font-size: 16px; color: lavender;'> <strong>Coeficiente de correlación (R):</strong> {coef_Correl:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 16px; color: lavender;'> <strong>Coeficiente de determinación (R²):</strong> {coef_Deter:.4f}</p>", unsafe_allow_html=True)
    else:
        st.warning("Por favor, escoge una variable dependiente y una independiente para que funcione :(")

#############################################################################################################
#Ahora definimos las funciones no lineales

def funcion_cuadratica(x, a, b, c):
    return a * x**2 + b * x + c

def funcion_exponencial(x, a, b, c):
    return a * np.exp(b * x) + c

def funcion_senoidal(x, a, b):
    return a * np.sin(x) + b

def funcion_tangencial(x, a, b):
    return a * np.tan(x) + b

def funcion_valor_absoluto(x, a, b, c):
    return a * np.abs(x) + b * x + c

def funcion_cociente_polinomios(x, a, b, c):
    return (a * x**2 + b) / (c * x)

def funcion_logaritmica(x, a, b):
    return a * np.log(x) + b

def funcion_polinomial_inversa(x, a, b, c):
    return a / b * x**2 + c * x

#Regresión no lineal
if Frames == "Regresion no lineal":
    #Título principal
    st.markdown("<h1 style='text-align: center; color: azure;'> Regresión no lineal </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: lighcyan;'>Selecciona el tipo de función para realizar el ajuste:</h2>", unsafe_allow_html=True)

    #Selección de variables
    x_selected3 = st.sidebar.selectbox(label="Escoge la variable x", options=numeric_cols)
    y_selected3 = st.sidebar.selectbox(label="Escoge la variable y", options=numeric_cols)

    #Selección de función de ajuste
    funcion_selected = st.sidebar.selectbox("Selecciona la función de ajuste",
        options=["Función cuadrática", "Función exponencial", "Función senoidal", "Función tangencial", "Función Valor absoluto", "Función cociente entre polinomios", "Función logaritmica", "Función polinomial inversa"])

    #Diccionario de funciones
    funciones = {
        "Función cuadrática": funcion_cuadratica,
        "Función exponencial": funcion_exponencial,
        "Función senoidal": funcion_senoidal,
        "Función tangencial": funcion_tangencial,
        "Función Valor absoluto": funcion_valor_absoluto,
        "Función cociente entre polinomios": funcion_cociente_polinomios,
        "Función logaritmica": funcion_logaritmica,
        "Función polinomial inversa": funcion_polinomial_inversa
    }

    #Obtener los datos seleccionados
    x3 = numeric_df[x_selected3].values
    y3 = numeric_df[y_selected3].values
    
    try:
        #Ajustar la función seleccionada a los datos escogidos
        popt, _ = curve_fit(funciones[funcion_selected], x3, y3)

        #Predecir los valores ajustados
        y_pred3 = funciones[funcion_selected](x3, *popt)

        #Gráfico de dispersión y la curva ajustada
        fig = px.scatter(x=x3, y=y3, title=f'Ajuste de {funcion_selected}',
                         labels={'x': x_selected3, 'y': y_selected3},color_discrete_sequence=["cyan"],
                         template='plotly_dark')

        #Agregamos la línea de ajuste
        fig.add_scatter(x=x3, y=y_pred3, mode='lines', name=f'Ajuste {funcion_selected}', line=dict(color='red', width=2))

        #Configuración del gráfico
        fig.update_layout(
            title_font=dict(size=20, color='thistle', family="Arial"),
            xaxis_title=dict(font=dict(size=16, color='thistle')),
            yaxis_title=dict(font=dict(size=16, color='thistle')),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
        )

        #Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)

        #Calcular el coeficiente de correlación
        correlacion3 = np.corrcoef(y3, y_pred3)[0, 1]

        #Mostrar el coeficiente de correlación con formato
        st.markdown(f"<p style='font-size: 18px; color: plum;'><strong>Coeficiente de correlación (R):</strong> {correlacion3:.4f}</p>", unsafe_allow_html=True)

    except Exception as e:
        #Mensaje de error mejorado
        st.error(f"No se ha podido ajustar la función: {str(e)}")
        st.markdown("<p style='color: red;'>Verifica que la variable seleccionada sea adecuada para la función seleccionada).</p>", unsafe_allow_html=True)

#############################################################################################################

#Boxplot
if Frames == "Boxplot":
    #Títulos y subtítulos más llamativos
    st.markdown("<h1 style='text-align: center; color: honeydew;'>Boxplot</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: linen;'>Correlaciones entre variables</h2>", unsafe_allow_html=True)
    
    # Crear un menú de selección en la Sidebar para que el usuario elija las columnas
    seleccion_columnas = st.sidebar.multiselect(
    'Selecciona las variables para el Boxplot:',
    options=numeric_cols,
    default=numeric_cols[:2]  # Selección predeterminada (las dos primeras columnas)
    )

# Verificar que el usuario haya seleccionado al menos una variable
    if seleccion_columnas:
        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(6, 8))  # Tamaño ajustado para mejor visualización

        # Generar el boxplot con las columnas seleccionadas
        sns.boxplot(data=df[seleccion_columnas], ax=ax)

        # Título del boxplot con formato mejorado
        ax.set_title("Boxplot de las Variables Seleccionadas", fontsize=18, color="#34495e", pad=20)

        # Ajustar los ticks y su estilo
        plt.xticks(fontsize=10, rotation=45, ha='right')  # Rota las etiquetas del eje x si es necesario
        plt.yticks(fontsize=10)  # Ajusta el tamaño de las etiquetas del eje y

        # Mostrar el boxplot en Streamlit
        st.pyplot(fig)
    else:
        st.write("Por favor, selecciona al menos una variable para mostrar el boxplot.")
