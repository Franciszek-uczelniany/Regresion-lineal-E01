# Importar librerías
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configuración de la aplicación
st.set_page_config(page_title="Predicción de Valores de Propiedades en Boston", layout="wide", page_icon="🏡")
st.title("🏡 Análisis y Predicción de Valores de Propiedades en Boston")
st.write("Esta aplicación permite analizar y predecir valores de propiedades en Boston utilizando un modelo de regresión lineal.")

# Cargar los datos
df = pd.read_csv('boston.csv').drop('Unnamed: 0', axis=1)

# Crear pestañas para la navegación
tabs = st.tabs(["📊 Datos", "📈 Modelo y Coeficientes", "📉 Evaluación del Modelo", "🔮 Simulación de Predicción"])

# Pestaña: Exploración de Datos
with tabs[0]:
    st.header("Exploración de Datos")
    st.write("Visualiza los primeros registros y las estadísticas descriptivas del conjunto de datos de Boston.")
    st.write(df.head())
    
    # Desplegable de estadísticas descriptivas
    with st.expander("Ver Estadísticas Descriptivas"):
        st.write(df.describe())

# Selección de variables independientes y dependientes
y = df.iloc[:, 13]
x = df.iloc[:, :-1]

# Pestaña: Modelo y Coeficientes
with tabs[1]:
    st.header("Modelo de Regresión Lineal y Coeficientes")
    
    # División en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=18231)
    
    # Entrenamiento del modelo
    modelo_1 = linear_model.LinearRegression()
    modelo_1.fit(x_train, y_train)
    
    # Mostrar coeficientes en un DataFrame
    coef_df = pd.DataFrame({"Característica": x.columns, "Coeficiente": modelo_1.coef_})
    st.write(coef_df)
    
    # Gráfico de coeficientes
    st.subheader("Gráfico de Coeficientes del Modelo")
    fig, ax = plt.subplots(figsize=(8, 4))
    coef_df.set_index("Característica").plot(kind="bar", ax=ax, color="lightcoral")
    ax.set_title("Coeficientes del Modelo")
    ax.set_ylabel("Valor del Coeficiente")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Pestaña: Evaluación del Modelo
with tabs[2]:
    st.header("Evaluación del Modelo")
    
    # Predicciones y métricas
    modelo_1_yhat = modelo_1.predict(x_test)
    mse = mean_squared_error(y_test, modelo_1_yhat)
    r2 = r2_score(y_test, modelo_1_yhat)
    
    # Presentación de métricas en columnas
    st.subheader("Métricas del Modelo")
    col1, col2 = st.columns(2)
    col1.metric("📉 Error Cuadrático Medio (MSE)", f"{mse:.2f}")
    col2.metric("📊 Coeficiente de Determinación (R²)", f"{r2:.2f}")
    
    # Gráfico de Predicciones vs Valores Reales
    st.subheader("Comparación entre Predicciones y Valores Reales")
    fig, ax = plt.subplots()
    ax.scatter(y_test, modelo_1_yhat, alpha=0.7, color="mediumseagreen")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Predicciones")
    ax.set_title("Predicciones vs Valores Reales")
    st.pyplot(fig)

# Pestaña: Simulación de Predicción
with tabs[3]:
    st.header("Simulación de Valor para Nuevos Vecindarios")
    st.write("Ingrese las características del vecindario para predecir su valor.")
    
    # Predicción para el peor vecindario
    worst_values = [0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]
    worst_neighbor = modelo_1.predict(np.array(worst_values).reshape(1, -1))
    st.info(f"Predicción para el peor vecindario: **${worst_neighbor[0]:,.2f}**")
    
    # Formulario interactivo para simulación
    with st.form(key="vecindario_form"):
        values = []
        for i, col in enumerate(x.columns):
            values.append(st.number_input(f"{col}", value=float(worst_values[i])))
        submit_button = st.form_submit_button(label="Predecir Valor")
    
        if submit_button:
            input_data = np.array(values).reshape(1, -1)
            prediccion = modelo_1.predict(input_data)
            st.success(f"Valor predicho para el vecindario ingresado: **${prediccion[0]:,.2f}**")
            
# Nota informativa
st.info("Esta aplicación proporciona un análisis basado en un modelo de regresión lineal y se enfoca en los factores más relevantes para predecir el valor de propiedades en Boston.")
