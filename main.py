import os
import sqlite3
import pickle as pc
import random
import time
import threading

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import compute_class_weight
from scipy.stats import norm

import threading
import time
import pymysql

# 1) Conexión SQLite en memoria:
conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row

# 2) ejecucion de SQL_sentencias
script_path = os.path.join(os.path.dirname(__file__), "SQL_sentencias.sql")
with open(script_path, "r", encoding="utf-8") as f:
    sql_script = f.read()
conn.executescript(sql_script)

# 3) Obtener datos quality
df_quality = pd.read_sql_query("SELECT * FROM quality", conn)

# 4) Obtener datos winequality-red
df = pd.read_sql_query("SELECT * FROM 'winequality-red';", conn)

# 5) Hacer el JOIN para armar df_data_join (igual que antes)
query = """
    SELECT wq.*, q.quality
      FROM 'winequality-red' AS wq
INNER JOIN quality AS q
        ON wq.id_quality = q.id_quality
"""
df_data_join = pd.read_sql_query(query, conn)

# Dividir en DataFrame para mostrar y para ML
df_display = df.copy()  # para Streamlit (vistas y gráficas)
df_ml = df.copy()  # para preprocesar y entrenar

conn.close()

# ——— PREPROCESAMIENTO Y ENTRENAMIENTO ———
# Mapear id_quality a 0–5 para modelado
df_ml['id_quality_mod'] = df_ml['id_quality'].map(
    {6: 0, 5: 1, 1: 2, 4: 3, 2: 4, 3: 5})

# Features y target
X = df_ml.drop(['id_quality', 'id_winequality', 'id_quality_mod'], axis=1)
y = df_ml['id_quality_mod']

# Calcular pesos inversos al tamaño de clase
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Entrenamiento
dt_regressor = DecisionTreeRegressor().fit(X_train, y_train)
xgb_model = XGBClassifier().fit(
    X_train, y_train, sample_weight=y_train.map(class_weights))

# Métricas
y_pred_dt = dt_regressor.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# Guardar modelo XGB si no existe
if not os.path.exists('pr.pkl'):
    pc.dump(xgb_model, open('pr.pkl', 'wb'))

# ——— INTERFAZ STREAMLIT ———
st.title("Análisis de la calidad de los vinos 🍷👌")

# Mostrar métricas de comparación
st.markdown("**Comparación de modelos entrenados:**")
st.write(f"- Decision Tree Regressor MSE: {mse_dt:.3f}")
st.write(f"- XGB Classifier Accuracy: {acc_xgb:.3%}")

# Vista previa de datos desde la base de datos
st.subheader("Vista previa de los datos (base de datos)")
st.dataframe(df_data_join.head())

# --------- Graficas ----------
# Distribución del contenido de alcohol
st.subheader("Distribución del contenido de alcohol (DB)")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df_display["alcohol"], kde=True, ax=ax1)
xticks = ax1.get_xticks()
ax1.set_xticks(xticks)
plt.xticks(rotation=45)
ax1.set_title("Distribución del Alcohol")
ax1.set_xlabel("Alcohol")
ax1.set_ylabel("Frecuencia")
st.pyplot(fig1)

# Distribución de la calidad del vino
st.subheader("Distribución de la calidad del vino (DB)")
fig2, ax2 = plt.subplots()
sns.countplot(x="quality", data=df_data_join, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.set_title("Frecuencia de id_quality")
ax2.set_xlabel("id_quality")
ax2.set_ylabel("Conteo")
st.pyplot(fig2)

# Promedio de ph de los vinos que son de calidad muy bueno

df_mb = df_display[df_display['id_quality'] == 4]
# 2. Calcular media, mediana y asimetría
mean_pH = df_mb['pH'].mean()
median_pH = df_mb['pH'].median()
skewness = df_mb['pH'].skew()
std_ph = df_mb['pH'].std()

# 3. Elegir estadístico según asimetría
if abs(skewness) > 0.5:
    value = median_pH
    label = 'Mediana'
else:
    value = mean_pH
    label = 'Media'
# 4. Plot
x = np.linspace(df_mb['pH'].min(), df_mb['pH'].max(), 200)
pdf = norm.pdf(x, loc=value, scale=std_ph)
# 4. Dibujar histograma normalizado y curva de Gauss
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_mb['pH'], bins=20, density=True, alpha=0.6, edgecolor='black')
ax.plot(x, pdf, linewidth=2)

st.subheader("Distribución de pH para vinos de categoria Muy buenos")
ax.set_title(
    f"Distribución de pH para vinos 'Muy Bueno' con curva gaussiana: (asimetria = {skewness:.2f})")
ax.set_xlabel("pH")
ax.set_ylabel("Densidad")
st.pyplot(fig)


# --------- FORMULARIO DE PREDICCIÓN ----------

st.header("Predicción de calidad de vino")

# Generar un valor aleatorio desde df_display
if "random_sample" not in st.session_state:
    st.session_state.random_sample = None

if st.button("🎲 Generar valores aleatorios"):
    st.session_state.random_sample = df_display.sample(1).squeeze()

# Usar valores aleatorios si existen, si no usar 0.0
sample_data = st.session_state.random_sample if st.session_state.random_sample is not None else {}

with st.form("wine_form"):
    fa = st.number_input(
        "Fixed Acidity",
        value=float(sample_data.get("fixed_acidity", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    va = st.number_input(
        "Volatile Acidity",
        value=float(sample_data.get("volatile_acidity", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    ca = st.number_input(
        "Citric Acid",
        value=float(sample_data.get("citric_acid", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    rs = st.number_input(
        "Residual Sugar",
        value=float(sample_data.get("residual_sugar", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    ch = st.number_input(
        "Chlorides",
        value=float(sample_data.get("chlorides", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    fs = st.number_input(
        "Free Sulfur Dioxide",
        value=float(sample_data.get("free_sulfur_dioxide", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    ts = st.number_input(
        "Total Sulfur Dioxide",
        value=float(sample_data.get("total_sulfur_dioxide", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    de = st.number_input(
        "Density",
        value=float(sample_data.get("density", 0.0)),
        min_value=0.0,
        step=0.0001,
        format="%.3f"
    )
    ph = st.number_input(
        "pH",
        value=float(sample_data.get("pH", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    su = st.number_input(
        "Sulphates",
        value=float(sample_data.get("sulphates", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
    al = st.number_input(
        "Alcohol",
        value=float(sample_data.get("alcohol", 0.0)),
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )

    submitted = st.form_submit_button("Predecir calidad")

if submitted:
    try:
        # Cargar modelo
        model = pc.load(open('pr.pkl', 'rb'))

        # Preparar DataFrame de entrada
        sample = pd.DataFrame([{
            "fixed_acidity":        fa,
            "volatile_acidity":     va,
            "citric_acid":          ca,
            "residual_sugar":       rs,
            "chlorides":            ch,
            "free_sulfur_dioxide":  fs,
            "total_sulfur_dioxide": ts,
            "density":              de,
            "pH":                   ph,
            "sulphates":            su,
            "alcohol":              al
        }])

        # Predicción y mapeo inverso
        pred_mod = model.predict(sample)[0]
        inverse_map = {0: 6, 1: 5, 2: 1, 3: 4, 4: 2, 5: 3}
        id_q = inverse_map[pred_mod]

        # Obtener etiqueta legible
        row = df_quality[df_quality['id_quality'] == id_q]
        label = row.iloc[0]['quality'] if not row.empty else str(id_q)

        st.success(f"La calidad estimada de tu vino es: **{label}**")

        # Mostrar input y predicción
        sample['Predicción'] = label
        st.dataframe(sample)

    except Exception as e:
        st.error("Error al cargar el modelo o hacer la predicción.")
        st.exception(e)

# --------- Perfil Quimico Promedio ----------
# Crear una copia para análisis, sin la columna id_quality
df_ml = df_display.copy()
# Aplicar mapeo correcto
df_ml['id_quality_mod'] = df_ml['id_quality'].map(
    {6: 0, 5: 1, 1: 2, 4: 3, 2: 4, 3: 5})
# Filtrar solo los vinos de calidad excepcional
df_excepcional = df_ml[df_ml['id_quality_mod'] == 5]
# Eliminar columnas irrelevantes
df_excepcional = df_excepcional.drop(
    columns=['id_quality', 'id_winequality', 'id_quality_mod'], errors='ignore')
# Calcular el promedio por columna (perfil ideal de vino excepcional)
mejores_valores = df_excepcional.mean().round(3)

# Mostrar resultados
st.subheader("📊 Perfil químico promedio de un vino excepcional")
st.write("Estos son los valores promedio de cada característica en vinos con calidad más alta:")
st.dataframe(mejores_valores.to_frame(name="Valor ideal").T)

# Botón para cargar el perfil promedio de vino excepcional en el formulario
if st.button("📥 Usar perfil promedio de vino excepcional"):
    st.session_state.random_sample = mejores_valores.to_dict()
    st.rerun()

# --------- Parametros que para el modelo son excepcionales ----------
# 1. Define el modelo y el mapeo
model = pc.load(open('pr.pkl', 'rb'))
inverse_map = {0: 6, 1: 5, 2: 1, 3: 4, 4: 2, 5: 3}

# 2. Rango de cada variable según df_display
features = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "pH", "sulphates", "alcohol"
]

df_rango = df_display.drop(
    columns=['id_quality', 'id_winequality'], errors='ignore')

ranges = {f: (df_rango[f].min(), df_rango[f].max()) for f in features}

# --- Generar solo una vez los parámetros excepcionales y guardarlos en session_state ---
if "best_excepcional" not in st.session_state:
    best = None
    for _ in range(20000):
        sample = {f: random.uniform(*ranges[f]) for f in features}
        df_sample = pd.DataFrame([sample])
        pred = model.predict(df_sample)[0]
        if pred == 5:  # 5 = excepcional según tu mapeo
            best = sample
            break
    st.session_state.best_excepcional = best
else:
    best = st.session_state.best_excepcional

if best:
    st.subheader("🔍 Parámetros que el modelo clasifica como Excepcional")
    out = pd.DataFrame([best]).round(3)
    st.dataframe(out)
    # Botón para cargar el perfil promedio de vino excepcional en el formulario
    if st.button("📥 Usar parámetros que el modelo clasifica como Excepcional"):
        st.session_state.random_sample = best
        st.rerun()
else:
    st.warning(
        "No encontramos en 20 000 intentos una combinación que clasifique como excepcional.")
