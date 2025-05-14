import pymysql

timeout = 10
connection = pymysql.connect(
  charset="utf8mb4",
  connect_timeout=timeout,
  cursorclass=pymysql.cursors.DictCursor,
  db="defaultdb",
  host="service-vinos-academia-c9e6.d.aivencloud.com",
  password=" ",
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