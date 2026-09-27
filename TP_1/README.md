# TP 1 — Regresión: predicción de precios de casas

Trabajo práctico de **Aprendizaje Automático 1**, Tecnicatura en Inteligencia Artificial
(FCEIA, UNR).

**Integrantes:** Calabozo · Darruiz · Di Carlo · Giuntoli

Construcción y comparación de modelos de **regresión lineal múltiple** para predecir `MEDV` (valor
mediano de las viviendas, en miles de dólares) a partir de 13 características del dataset de precios
de casas de Boston.

Todo el trabajo está en un único notebook, [`TP-regresion-AA1.ipynb`](TP-regresion-AA1.ipynb),
que funciona como informe: intercala celdas de código con bloques de texto que desarrollan el
análisis, justifican cada decisión con una métrica o un gráfico, y cierran con las conclusiones.
Las figuras están numeradas para poder referenciarlas desde el texto.

**Contenido del notebook**

1. Análisis descriptivo de las variables: rango, distribución y valores atípicos.
2. Tratamiento de los datos faltantes.
3. Relación entre las variables: correlación con el precio y entre características.
4. Preparación: división en entrenamiento y prueba (80/20) y `Pipeline` de imputación + escalado.
5. Modelado: `LinearRegression`, descenso por gradiente (batch, estocástico y mini-batch) y
   regularización (Ridge, Lasso, Elastic Net).
6. Optimización de hiperparámetros con **validación cruzada de 5 particiones** (`GridSearchCV`),
   comparación de modelos y conclusiones.

**Resultados principales**

- `LinearRegression` y los tres métodos de descenso por gradiente convergen a la misma solución,
  porque minimizan la misma función de costo convexa: sus coeficientes no se apartan más de 0.07
  entre sí. Se diferencian en la velocidad de convergencia y en la forma de las curvas de error.
- **Todos los modelos lineales resultan equivalentes** en este dataset. El mejor por validación
  cruzada es Ridge con `alpha ≈ 17.8` (RMSE 5.813 k\$), pero la mejora sobre `LinearRegression`
  (5.829 k\$) es de 0.016 k\$ contra un desvío entre particiones de 1.1 k\$.
- El caso de **Lasso con `alpha = 1`**, que es el mejor en prueba (RMSE 6.51 k\$) y el peor en
  validación cruzada (6.48 k\$), ilustra por qué no hay que elegir modelos mirando el conjunto de
  prueba.
- Sin limitarse a modelos lineales (`mejor-modelo.ipynb`), un ensamble de `ExtraTrees`,
  `GradientBoosting` y `SVR` baja el RMSE de prueba de 7.61 k\$ a 4.60 k\$ y sube el R² de 0.30 a
  0.75.

## Estructura

```
TP_1/
├── README.md
├── TP-regresion-AA1.ipynb                    Notebook principal (informe completo)
├── anexo-preprocesamiento.ipynb              Anexo: análisis del preprocesamiento
├── mejor-modelo.ipynb                        Búsqueda del mejor modelo, sin limitarse a lineales
├── data/
│   └── house-prices-tp.csv                   Dataset
├── src/
│   └── descenso_gradiente.py                 Implementaciones de descenso por gradiente
└── referencia/
    ├── consigna-TP1.pdf                      Consigna del trabajo práctico
    └── implementaciones-descenso-gradiente-catedra.ipynb   Notebook de la cátedra
```

El módulo `src/descenso_gradiente.py` contiene las implementaciones de **Gradient Descent**,
**Stochastic Gradient Descent** y **Mini-Batch Gradient Descent** tomadas de la notebook de la
cátedra, adaptadas para aceptar DataFrames de pandas y devolver el historial de error de
entrenamiento y validación, de modo que los tres métodos se puedan comparar en un mismo gráfico.

## Cómo ejecutar

Primero hay que levantar el entorno como se explica en el [README principal](../README.md). Después
se abre `TP-regresion-AA1.ipynb` con el kernel de `.venv`.

El notebook lee `data/house-prices-tp.csv` e importa `src/descenso_gradiente.py` con **rutas
relativas**, así que necesita que el directorio de trabajo sea `TP_1/`. Al abrirlo con Jupyter o
VS Code esto ya ocurre solo, porque el kernel arranca en la carpeta del notebook.

Está configurado con una semilla fija (`SEMILLA = 42`) en todas las divisiones de datos, las
inicializaciones de pesos y los gráficos con componente aleatoria, por lo que los resultados son
reproducibles: con las mismas versiones de las bibliotecas, ejecutarlo de nuevo devuelve exactamente
los mismos números que figuran en el texto.
