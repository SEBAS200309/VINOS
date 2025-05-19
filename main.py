import pymysql
import os
import pandas as pd

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
    


    print(df)

finally:
    connection.close()
