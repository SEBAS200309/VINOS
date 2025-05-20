import pymysql
import os
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
finally:
    connection.close()
