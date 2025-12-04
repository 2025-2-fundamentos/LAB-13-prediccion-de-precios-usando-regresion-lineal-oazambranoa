#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pickle
from glob import glob
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error,
)


def load_data(train_path="files/input/train_data.csv.zip",
              test_path="files/input/test_data.csv.zip"):

    df_train = pd.read_csv(train_path, compression="zip", index_col=False)
    df_test = pd.read_csv(test_path, compression="zip", index_col=False)
    return df_train, df_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Age"] = 2021 - data["Year"]
    data.drop(columns=["Year", "Car_Name"], inplace=True)
    return data


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return X, y


def make_pipeline(x_train: pd.DataFrame) -> Pipeline:
    cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    num_cols = [col for col in x_train.columns if col not in cat_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(), cat_cols),
        ("numerical", MinMaxScaler(), num_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression())
    ])

    return pipeline


def optimize_parameters(pipeline, x_train, y_train):
    n_total_features = pipeline.named_steps["preprocessor"].fit_transform(x_train).shape[1]

    param_grid = {
        "selector__k": range(1, n_total_features + 1),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid.fit(x_train, y_train)
    return grid


def clean_model_directory(path="files/models/"):
    if os.path.exists(path):
        for file in glob(f"{path}/*"):
            os.remove(file)
        os.rmdir(path)

    os.makedirs(path, exist_ok=True)


def save_model(model, path="files/models/model.pkl.gz"):
    clean_model_directory(os.path.dirname(path))

    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def calculate_metrics(dataset_type, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "r2": round(r2_score(y_true, y_pred), 4),
        "mse": round(mean_squared_error(y_true, y_pred), 4),
        "mad": round(median_absolute_error(y_true, y_pred), 4),
    }


def save_metrics(metrics_list, path="files/output/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")


# --- Ejecución principal ---
train_df, test_df = load_data()

train_df = clean_data(train_df)
test_df = clean_data(test_df)

X_train, y_train = split_data(train_df)
X_test, y_test = split_data(test_df)

pipeline = make_pipeline(X_train)
best_estimator = optimize_parameters(pipeline, X_train, y_train)

save_model(best_estimator)

y_pred_train = best_estimator.predict(X_train)
y_pred_test = best_estimator.predict(X_test)

metrics_out = [
    calculate_metrics("train", y_train, y_pred_train),
    calculate_metrics("test", y_test, y_pred_test)
]

save_metrics(metrics_out)
