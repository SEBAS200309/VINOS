import os
import pymysql
import pickle as pc

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# ——— CONFIGURACIÓN DE BASE DE DATOS ———
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

# ——— CARGA DE DATOS Y ENTRENAMIENTO ———
with connection.cursor() as cursor:
    # Datos de vinos
    cursor.execute("SELECT * FROM defaultdb.`winequality-red`")
    results = cursor.fetchall()
    df = pd.DataFrame(results)

    # Tabla de calidad para mostrar nombres
    cursor.execute("SELECT * FROM defaultdb.quality")
    stats = cursor.fetchall()
    df_quality = pd.DataFrame(stats)

# Preprocesamiento
# Mapear id_quality a 0–5 para modelado
df['id_quality_mod'] = df['id_quality'].map({6:0, 5:1, 1:2, 4:3, 2:4, 3:5})
X = df.drop(['id_quality', 'id_winequality', 'id_quality_mod'], axis=1)
y = df['id_quality_mod']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Entrenar modelos
dt_regressor = DecisionTreeRegressor().fit(X_train, y_train)
xgb_model    = XGBClassifier().fit(X_train, y_train)

# Métricas de comparación
y_pred_dt  = dt_regressor.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
mse_dt  = mean_squared_error(y_test, y_pred_dt)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# Guardar modelo de ejemplo (solo si no existe)
if not os.path.exists('pr.pkl'):
    pc.dump(xgb_model, open('pr.pkl','wb'))

# Cerrar conexión
connection.close()


# ——— INTERFAZ STREAMLIT ———
st.title("Análisis de la calidad de los vinos 🍷👌")

st.markdown("**Comparación de modelos entrenados:**")
st.write(f"- Decision Tree Regressor MSE: {mse_dt:.3f}")
st.write(f"- XGB Classifier Accuracy: {acc_xgb:.3%}")

# Vista previa de datos (desde CSV local)
st.subheader("Vista previa de los datos CSV")
df_csv = pd.read_csv("vinos_1 .csv", sep=";")
st.dataframe(df_csv.head())

# Gráficos
st.subheader("Distribución del contenido de alcohol")
fig1, ax1 = plt.subplots()
sns.histplot(df_csv["alcohol"], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("Distribución de la calidad del vino")
fig2, ax2 = plt.subplots()
sns.countplot(x="quality", data=df_csv, ax=ax2)
st.pyplot(fig2)


# Formulario de predicción
st.header("Predicción de calidad de vino")
with st.form("wine_form"):
    fa = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
    va = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
    ca = st.number_input("Citric Acid", min_value=0.0, step=0.01)
    rs = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
    ch = st.number_input("Chlorides", min_value=0.0, step=0.001)
    fs = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
    ts = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
    de = st.number_input("Density", min_value=0.0, step=0.0001)
    ph = st.number_input("pH", min_value=0.0, step=0.01)
    su = st.number_input("Sulphates", min_value=0.0, step=0.01)
    al = st.number_input("Alcohol", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("Predecir calidad")

if submitted:
    try:
        # Cargar modelo entrenado
        model = pc.load(open('pr.pkl', 'rb'))

        # Crear DataFrame de input con los nombres de columna que el modelo espera
        sample = pd.DataFrame([{
            "fixed_acidity": fa,
            "volatile_acidity": va,
            "citric_acid": ca,
            "residual_sugar": rs,
            "chlorides": ch,
            "free_sulfur_dioxide": fs,
            "total_sulfur_dioxide": ts,
            "density": de,
            "pH": ph,
            "sulphates": su,
            "alcohol": al
        }])

        # Predecir y mapear de vuelta a id_quality original
        pred_mod = model.predict(sample)[0]
        inverse_map = {0:6, 1:5, 2:1, 3:4, 4:2, 5:3}
        id_q = inverse_map[pred_mod]

        # Obtener nombre de calidad
        row = df_quality[df_quality['id_quality']==id_q]
        label = row.iloc[0]['quality'] if not row.empty else str(id_q)

        st.success(f"La calidad estimada de tu vino es: **{label}**")

        # Mostrar tabla con input y resultado
        sample['Predicción'] = label
        st.dataframe(sample)

    except Exception as e:
        st.error("Error al cargar el modelo o hacer la predicción.")
        st.exception(e)
