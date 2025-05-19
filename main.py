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

    df = pd.DataFrame(results)
    df.reset_index()

    df['id_quality'].replace({ 3 : 5, 2: 4, 4: 3,1: 2, 5: 1, 6: 0}, inplace=True)
    x = df.drop('id_quality', axis=1)
    y = df['id_quality']
    y
    print("Categorias de consumo: \n", y.value_counts())
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    x_test
    y_test
    
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(x_train, y_train)
    y_pred_dt = dt_regressor.predict(x_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    print('Mean Squared Error (Decision Tree):', mse_dt)
    
    print('Comparacion Modelos: \n')
    print('Mean Squared Error (Decision Tree):', mse_dt)
    print('Accuracy (XGB):', accuracy_dt)

    dtclae_dt = XGBClassifier()
    dtclae_dt.fit(x_train, y_train)
    y_pred_dt = dtclae_dt.predict(x_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print('Accuracy (XGB):', accuracy_dt)

    print(df)

finally:
    connection.close()
