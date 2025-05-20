import pymysql
import os
import streamlit as st
import pandas as pd
import pickle as pc
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

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
  cursor.execute("SELECT * FROM defaultdb.`winequality-red` LIMIT 10")
  print(cursor.fetchall())
finally:
  connection.close()

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