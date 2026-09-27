"""
Implementaciones de descenso del gradiente para regresión lineal.

Funciones tomadas de la notebook de la cátedra
(referencia/implementaciones-descenso-gradiente-catedra.ipynb), con las
siguientes adaptaciones para poder reutilizarlas desde TP-regresion-AA1.ipynb:

- Aceptan DataFrames/Series de pandas además de arrays de NumPy.
- Devuelven, además de los pesos W, el historial de MSE de entrenamiento y
  de validación, para poder comparar los métodos en un mismo gráfico.
- En los tres métodos el MSE se registra una vez por época y sobre el
  conjunto completo, así las curvas de GD, SGD y Mini-Batch son comparables
  entre sí (loss vs epochs).
- El gráfico es opcional (parámetro `graficar`), y se le puede cambiar el
  título (`titulo`) y la etiqueta de la segunda curva (`etiqueta_val`).
- Se puede fijar una `semilla` para que los resultados sean reproducibles.
- En Mini-Batch el gradiente se divide por el tamaño real del lote (el
  último lote puede tener menos de `batch_size` muestras).

Todos los modelos son de la forma y_pred = X @ W, donde X incluye una
columna de unos para el término independiente (bias), y minimizan el Error
Cuadrático Medio (MSE).
"""

import numpy as np
import matplotlib.pyplot as plt


def _agregar_bias(X):
    """Convierte X a array de NumPy y le agrega una columna de unos (bias)."""
    X = np.asarray(X, dtype=float)
    return np.hstack((np.ones((X.shape[0], 1)), X))


def _como_columna(y):
    """Convierte y a un array de NumPy de dimensiones (n, 1)."""
    return np.asarray(y, dtype=float).reshape(-1, 1)


def _mse(X, y, W):
    """MSE de las predicciones X @ W respecto de y (X ya incluye el bias)."""
    return np.mean((y - np.matmul(X, W)) ** 2)


def predecir(X, W):
    """
    Predice con los pesos W obtenidos por cualquiera de los métodos.

    Parámetros
    ----------
    X : np.ndarray o pd.DataFrame
        Matriz de características de dimensiones (n, m), sin columna de bias.
    W : np.ndarray
        Vector de pesos de dimensiones (m + 1, 1); W[0] es el bias.

    Retorna
    -------
    np.ndarray
        Vector de predicciones de dimensiones (n,).
    """
    return np.matmul(_agregar_bias(X), W).ravel()


def graficar_errores(train_errors, val_errors, titulo,
                     label_val='Error de validación'):
    """Grafica la evolución del MSE de entrenamiento y validación por época."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_errors, label='Error de entrenamiento')
    plt.plot(val_errors, label=label_val)
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio')
    plt.legend()
    plt.title(titulo)
    plt.show()


def gradient_descent(X_train, y_train, X_val, y_val, lr=0.01, epochs=100,
                     semilla=None, graficar=True, titulo=None,
                     etiqueta_val='Error de validación'):
    """
    Entrena un modelo de regresión lineal mediante Gradient Descent (batch).

    En cada época se calcula el gradiente utilizando todas las muestras de
    entrenamiento y se realiza una única actualización de los pesos.

    Parámetros
    ----------
    X_train : np.ndarray o pd.DataFrame
        Matriz de características de entrenamiento de dimensiones (n, m).
    y_train : np.ndarray o pd.Series
        Valores objetivo de entrenamiento (n valores).
    X_val : np.ndarray o pd.DataFrame
        Matriz de características de validación de dimensiones (p, m).
    y_val : np.ndarray o pd.Series
        Valores objetivo de validación (p valores).
    lr : float, optional
        Tasa de aprendizaje. Por defecto 0.01.
    epochs : int, optional
        Cantidad de épocas (= cantidad de actualizaciones). Por defecto 100.
    semilla : int, optional
        Semilla para la inicialización aleatoria de los pesos.
    graficar : bool, optional
        Si es True, grafica el MSE de entrenamiento y validación por época.
    titulo : str, optional
        Título del gráfico. Por defecto, el de la notebook de la cátedra.
    etiqueta_val : str, optional
        Etiqueta de la curva de error de X_val, y_val en el gráfico.

    Retorna
    -------
    W : np.ndarray
        Vector de pesos de dimensiones (m + 1, 1), incluyendo el bias en W[0].
    train_errors : list[float]
        MSE de entrenamiento al final de cada época.
    val_errors : list[float]
        MSE de validación al final de cada época.

    Notas
    -----
    El gradiente del MSE respecto de W es:

        ∇J(W) = -(2 / n) X.T (y - XW)

    y la actualización de los pesos es W = W - lr * ∇J(W).
    """
    rng = np.random.default_rng(semilla)

    # Poner columna de unos a las matrices X y llevar y a vectores columna
    X_train, X_val = _agregar_bias(X_train), _agregar_bias(X_val)
    y_train, y_val = _como_columna(y_train), _como_columna(y_val)
    n, m = X_train.shape

    # Inicializar pesos aleatorios
    W = rng.standard_normal((m, 1))

    train_errors = []  # MSE de entrenamiento en cada época
    val_errors = []    # MSE de validación en cada época

    for _ in range(epochs):
        # Calcular el gradiente con todo el conjunto de entrenamiento y
        # actualizar pesos
        error = y_train - np.matmul(X_train, W)
        gradient = -2 / n * np.matmul(X_train.T, error)
        W = W - lr * gradient

        # Registrar errores de entrenamiento y validación
        train_errors.append(_mse(X_train, y_train, W))
        val_errors.append(_mse(X_val, y_val, W))

    if graficar:
        graficar_errores(
            train_errors, val_errors,
            titulo or 'Error de entrenamiento y validación vs épocas (GD)',
            etiqueta_val)

    return W, train_errors, val_errors


def stochastic_gradient_descent(X_train, y_train, X_val, y_val, lr=0.01,
                                epochs=100, semilla=None, graficar=True,
                                titulo=None,
                                etiqueta_val='Error de validación'):
    """
    Entrena una regresión lineal mediante Stochastic Gradient Descent (SGD).

    A diferencia de Gradient Descent, que calcula el gradiente con todas las
    muestras antes de actualizar los pesos, SGD actualiza los pesos después
    de procesar cada muestra individual (n actualizaciones por época).

    Parámetros
    ----------
    X_train, y_train, X_val, y_val, lr, epochs, semilla, graficar, titulo,
    etiqueta_val :
        Igual que en `gradient_descent`.

    Retorna
    -------
    W, train_errors, val_errors :
        Igual que en `gradient_descent`. Los errores se calculan sobre los
        conjuntos completos al final de cada época.

    Notas
    -----
    En cada época se permutan aleatoriamente las muestras de entrenamiento
    para evitar que el orden de los datos afecte al aprendizaje.

    El gradiente para una muestra individual (x_i, y_i) es:

        ∇J(W) = -2 * x_i.T * (y_i - x_i @ W)
    """
    rng = np.random.default_rng(semilla)

    X_train, X_val = _agregar_bias(X_train), _agregar_bias(X_val)
    y_train, y_val = _como_columna(y_train), _como_columna(y_val)
    n, m = X_train.shape

    W = rng.standard_normal((m, 1))

    train_errors = []
    val_errors = []

    for _ in range(epochs):
        # Permutación aleatoria de los datos
        permutation = rng.permutation(n)
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        for j in range(n):
            # Tomar una única muestra, calcular su gradiente y actualizar
            # los pesos
            x_sample = X_train[j:j + 1]  # (1, m)
            y_sample = y_train[j:j + 1]  # (1, 1)

            error = y_sample - np.matmul(x_sample, W)
            gradient = -2 * np.matmul(x_sample.T, error)
            W = W - lr * gradient

        # Registrar errores al final de la época
        train_errors.append(_mse(X_train, y_train, W))
        val_errors.append(_mse(X_val, y_val, W))

    if graficar:
        graficar_errores(
            train_errors, val_errors,
            titulo or 'Error de entrenamiento y validación vs épocas (SGD)',
            etiqueta_val)

    return W, train_errors, val_errors


def mini_batch_gradient_descent(X_train, y_train, X_val, y_val, lr=0.01,
                                epochs=100, batch_size=11, semilla=None,
                                graficar=True, titulo=None,
                                etiqueta_val='Error de validación'):
    """
    Entrena un modelo de regresión lineal mediante Mini-Batch Gradient Descent.

    Combina Gradient Descent y SGD: en lugar de calcular el gradiente con todo
    el conjunto de entrenamiento o con una única muestra, lo calcula con
    pequeños lotes (mini-batches) de `batch_size` muestras.

    Parámetros
    ----------
    X_train, y_train, X_val, y_val, lr, epochs, semilla, graficar, titulo,
    etiqueta_val :
        Igual que en `gradient_descent`.
    batch_size : int, optional
        Cantidad de muestras utilizadas en cada actualización. Por defecto 11.

    Retorna
    -------
    W, train_errors, val_errors :
        Igual que en `gradient_descent`. Los errores se calculan sobre los
        conjuntos completos al final de cada época.

    Notas
    -----
    El gradiente para un lote (X_batch, y_batch) es:

        ∇J(W) = -(2 / len(X_batch)) * X_batch.T @ (y_batch - X_batch @ W)

    Según el tamaño del lote:
        - Gradient Descent: utiliza todas las muestras.
        - Stochastic Gradient Descent: utiliza una única muestra.
        - Mini-Batch Gradient Descent: utiliza un pequeño grupo de muestras.
    """
    rng = np.random.default_rng(semilla)

    X_train, X_val = _agregar_bias(X_train), _agregar_bias(X_val)
    y_train, y_val = _como_columna(y_train), _como_columna(y_val)
    n, m = X_train.shape

    W = rng.standard_normal((m, 1))

    train_errors = []
    val_errors = []

    for _ in range(epochs):
        # Permutación aleatoria de los datos
        permutation = rng.permutation(n)
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        for j in range(0, n, batch_size):
            # Obtener un lote (mini-batch), calcular su gradiente y
            # actualizar los pesos
            x_batch = X_train[j:j + batch_size]
            y_batch = y_train[j:j + batch_size]

            error = y_batch - np.matmul(x_batch, W)
            gradient = -2 * np.matmul(x_batch.T, error) / len(x_batch)
            W = W - lr * gradient

        # Registrar errores al final de la época
        train_errors.append(_mse(X_train, y_train, W))
        val_errors.append(_mse(X_val, y_val, W))

    if graficar:
        graficar_errores(
            train_errors, val_errors,
            titulo or ('Error de entrenamiento y validación vs épocas '
                       '(Mini-Batch GD)'),
            etiqueta_val)

    return W, train_errors, val_errors
