import pymysql
import os

import streamlit as st
import pickle as pc
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

timeout = 10
connection = pymysql.connect(
    charset="utf8mb4",
    connect_timeout=timeout,
    cursorclass=pymysql.cursors.DictCursor,
    db="defaultdb",
    host="service-vinos-academia-c9e6.d.aivencloud.com",
    password=os.getenv("DB_PASSWORD"),
    read_timeout=timeout,
    port=11434,
    user="avnadmin",
    write_timeout=timeout,
)

try:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM defaultdb.`winequality-red`")
    results = cursor.fetchall()

    cursor.execute ("SELECT * FROM defaultdb.quality")
    stats = cursor.fetchall()

    df_s = pd.DataFrame(stats)
    print (df_s)

    df = pd.DataFrame(results)
    df.reset_index()

    df['id_quality'].replace(
        {3: 5, 2: 4, 4: 3, 1: 2, 5: 1, 6: 0}, inplace=True)
    x = df.drop(['id_quality', 'id_winequality'], axis=1)
    y = df['id_quality']


    print("Categorias de consumo: \n", y.value_counts())

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    x_test
    y_test

    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(x_train, y_train)
    y_pred_dt = dt_regressor.predict(x_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)

    xgbModel = XGBClassifier()
    xgbModel.fit(x_train, y_train)
    y_pred_dt = xgbModel.predict(x_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)

    print('\nComparacion Modelos: ')
    print('Mean Squared Error (Decision Tree):', mse_dt)
    print('Accuracy (XGB):', accuracy_dt)

    # Valores de ejemplo (tal como los indicaste)
    sample = {
        'fixed_acidity':            [8.4],
        'volatile_acidity':         [0.500],
        'citric_acid':              [0.01],
        'residual_sugar':           [1.5],
        'chlorides':                [0.07],
        'free_sulfur_dioxide':      [9.0],
        'total_sulfur_dioxide':     [25.0],
        'density':                  [0.99780],
        'pH':                       [3.4],
        'sulphates':                [0.7],
        'alcohol':                  [11.4]
    }

    df_sample = pd.DataFrame(sample)


    # 3.2 Con el clasificador XGB
    pred_clf = xgbModel.predict(df_sample)

    # Invertir el mapeo para obtener el id_quality original
    inverse_mapping = {5: 3, 4: 2, 3: 4, 2: 1, 1: 5, 0: 6}
    pred_class = int(pred_clf[0])
    original_id_quality = inverse_mapping.get(pred_class)

    if original_id_quality is not None:
        # Buscar el nombre de la categoría en el DataFrame df_s
        categoria_row = df_s[df_s['id_quality'] == original_id_quality]
        if not categoria_row.empty:
            categoria_nombre = categoria_row.iloc[0]['quality']
            prediccion = (f"El vino en cuestion tendra una calidad: {categoria_nombre}")
            print(prediccion)
        else:
            print("No se encontró la categoría en la base de datos.")
    else:
        print(f"No se pudo mapear la clase {pred_class} predicha al id_quality original.")

st.title("Analisis de la calidad de los vinos 🍷👌")


file_path = "vinos_1 .csv"
df = pd.read_csv(file_path, sep=";")

# Vista previa
st.subheader("Vista previa de los datos")
st.dataframe(df.head())


st.subheader("Distribucion del contenido de alcohol")

plt.figure(figsize=(10, 5))
sns.histplot(df["alcohol"], kde=True, color="crimson", bins=30)
plt.title("Distribución del Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Frecuencia")
xticks = plt.xticks()[0]                     
filtered = xticks[::4]                        
plt.xticks(filtered, rotation=45)             
st.pyplot(plt)

st.subheader("Distribucion de la calidad del vino (Diagrama de barras)")

plt.figure(figsize=(8, 4))
sns.countplot(x="quality", data=df, palette="viridis")
plt.title("Frecuencia de calidad del vino")
plt.xlabel("Calidad")
plt.ylabel("Cantidad de muestras")
st.pyplot(plt)





st.title("Prediccion de calidad de vinos 🍷👌")



st.header("Ingresa las caracteristicas de tu vino para predecir su calidad")
st.markdown("Completa los siguientes campos con los valores correspondientes. Todos los datos deben ser numericos:")


with st.form("wine_form"):
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
    density = st.number_input("Density", min_value=0.0, step=0.0001)
    pH = st.number_input("pH", min_value=0.0, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
    alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predecir calidad")


if submitted:
    try:
        # Cargar modelo previamente entrenado
        load_pred = pc.load(open('pr.pkl', 'rb'))

        # Crear DataFrame con los datos ingresados
        input_data = pd.DataFrame([{
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_sulfur_dioxide,
            "total sulfur dioxide": total_sulfur_dioxide,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol
        }])

        # Realizar predicción
        prediction = load_pred.predict(input_data)[0]
        st.success(f"La calidad estimada de tu vino es: **{prediction}** 🎯")

        # Mostrar tabla de entrada + predicción
        input_data["Predicción de Calidad"] = prediction
        st.dataframe(input_data)

    except Exception as e:
        st.error("Hubo un error al cargar el modelo o al hacer la predicción.")
        st.exception(e)
    finally:
      
    connection.close()
