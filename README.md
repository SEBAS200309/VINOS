# 🍷 Análisis de Calidad de Vinos con Streamlit

Este proyecto es una aplicación web interactiva que:

1. Conecta a una base de datos MySQL remota para cargar datos de vino tinto y una tabla de calidad.  
2. Entrena dos modelos (Decision Tree Regressor y XGBoost Classifier con pesos de clase balanceados) sobre esos datos.  
3. Presenta comparativas de rendimiento y visualizaciones de distribución.  
4. Permite al usuario generar muestras aleatorias (o “ideales”) y predecir la calidad de un vino con un formulario de entrada.  
5. Incluye secciones para mostrar el perfil químico promedio de vinos excepcionales y buscar parámetros que el modelo clasifica como “Excepcional”.

---

## 📦 Estructura del proyecto

```css
├── .devcontainer/ # Configuración de VS Code + Docker para desarrollo
├── Dockerfile # Imagen base para construir y ejecutar la app
├── requirements.txt # Todas las dependencias de Python
├── main.py # Código principal de Streamlit
├── pr.pkl # Modelo XGBoost serializado (generado en la primera ejecución)
└── README.md # Este documento
```

---

## 🔧 Prerrequisitos

- Docker ≥ 20.10  
- VS Code con la extensión **“Remote – Containers”** (opcional)  
- Cuenta de GitHub con colaborador en el repositorio (para acceder al secreto `DB_PASSWORD`)

---

## 🚀 Desarrollo local con devcontainer

1. **Clona** este repositorio:
   ```bash
   git clone git@github.com:TuUsuario/tu-repo-vinos.git
   cd tu-repo-vinos

---

## 🔧 Prerrequisitos

- Docker ≥ 20.10  
- VS Code con la extensión **“Remote – Containers”** (opcional)  
- Cuenta de GitHub con colaborador en el repositorio (para acceder al secreto `DB_PASSWORD`)

---

## 🚀 Desarrollo local con devcontainer

1. **Clona** este repositorio:
   ```bash
   git clone git@github.com:TuUsuario/tu-repo-vinos.git
   cd tu-repo-vinos
2. Abre VS Code en esta carpeta. Aparecerá un mensaje para “Reopen in Container”. Acepta y espera a que Docker cree el entorno.

3. El contenedor:

* Instalará todas las librerías de requirements.txt.

* Tendrá acceso a tu variable DB_PASSWORD a través de devcontainer.json (ver más abajo).

4. Una vez abierto el container, ejecuta:
```bash
streamlit run main.py
```

## 🏗️ Configuración de .devcontainer/devcontainer.json
```json
{
  "name": "Streamlit Vinos",
  "build": {
    "dockerfile": "../Dockerfile"
  },
  "remoteUser": "root",
  "remoteEnv": {
    "DB_PASSWORD": "${localEnv:DB_PASSWORD}"
  }
}
```

## 📑 Uso de la aplicación
Al iniciar la app verás:

1. Comparativa de modelos: MSE del árbol vs. Accuracy de XGBoost.

2. Vista previa de los datos SQL unidos con la tabla de calidad.

3. Gráficas de distribución de alcohol y calidad.

4. Campana de Gauss sobre la distribución de pH para “Muy Bueno”.

5. Formulario para predecir calidad:

    * Botón 🎲 para obtener valores aleatorios reales.
    * Hasta 3 decimales en cada campo.

6. Sección de perfil químico promedio de vinos excepcionales.

7. Búsqueda aleatoria de parámetros que el modelo clasifica como “Excepcional”.

## 📈 Cómo se entrena el modelo
* Se lee el dataset completo en df_ml.

* Se mapea id_quality a orden ordinal 0–5 (0 = “defectuoso”, 5 = “excepcional”).

* Se calculan class_weights inversos al desequilibrio y se entrena XGBClassifier con sample_weight.

* El modelo resultante se guarda como pr.pkl la primera vez que corre.

## 📚 Referencias

* [Streamlit Docs] (https://docs.streamlit.io)
* [XGBoost python API] (https://xgboost.readthedocs.io/en/release_3.0.0/)
* [pymsql docs] (https://pymysql.readthedocs.io/en/latest/)
