import pymysql
import os
import streamlit as st
import pandas as pd
import pickle as pc
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



st.title("Prediccion calidad de vinos🍷👌")
st.write('------')
st.header("Cargue un archivo csv con las variables Turbidez, Ph y caudal")
st.subheader("Prediccion de vinos con el ph de mejor calidad👌")
st.subheader("introducir grafico")
st.write("Explicacion del grafico blablblabablablablablablablablablablablablablbalblablablablablahfidghhfkasyfigfjtfjgfjkjdgsjfvuash")
st.subheader("Tipos de calidad de vino🍷")
st.subheader("introducir grafico")
st.write("Explicacion del grafico blablblabablablablablablablablablablablablablbalblablablablablahfidghhfkasyfigfjtfjgfjkjdgsjfvuash")
st.header("¿Quieres saber la calidad que podria tener tu vino?😉")

st.slider("hola",0,100,10)
uploaded_file = st.file_uploader("Cargue su archivo csv" ,type=["csv"])

if uploaded_file is not None:
    input_dfd = pd.read_csv(uploaded_file,sep=";")

    load_pred = pc.load(open('pr.pkl', 'rb'))
    input_dfd['estimacion_efi'] = load_pred.predict(input_dfd)
    st.write(input_dfd)