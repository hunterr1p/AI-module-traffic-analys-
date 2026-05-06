from google.colab import drive
drive.mount('/content/drive')


import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

#настройка
#папка с файлами parquest/csv

DATASET_FOLDER = "/content/drive/MyDrive/filesddos"

#папка сохр модели и результаты
RESULTS_FOLDER = "/content/drive/MyDrive/filesddos/results"

#метки
#true - все метки кроме BENIGN ,будут считаться ATTACK
#false - оставит исходные метки как есть (multiclass)

BINARY_MODE = True

TEST_SIZE = 0.2

#рандом сид

RANDOM_STATE = 42

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("DATASET_FOLDER =", DATASET_FOLDER)
print("RESULTS_FOLDER =", RESULTS_FOLDER)


def load_dataset(folder_path):
    parquet_files = []
    csv_files = []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)

        if file.endswith("training.parquet"):
            parquet_files.append(full_path)

        elif file.endswith("training.csv"):
            csv_files.append(full_path)

    files = parquet_files + csv_files

    if len(files) == 0:
        raise Exception("Не найдено файлов training.parquet или training.csv")

    print("Найдено файлов:", len(files))

    dataframes = []

    for file in files:
        print("Загрузка:", file)

        if file.endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file, low_memory=False)

        dataframes.append(df)

    dataset = pd.concat(dataframes, ignore_index=True)

    print("Датасет загружен")
    print("Размер:", dataset.shape)

    return dataset

def normalize_labels(y):
    if not BINARY_MODE:
        return y.astype(str)


    y = y.astype(str).str.strip()
    y = y.apply(lambda x: "BENIGN" if x.upper() == "BENIGN" else "ATTACK")
    return y

def prepare_data(dataset):
    print("Подготовка данных...")

    if "Label" not in dataset.columns:
        raise Exception("Колонка Label не найдена")

    df = dataset.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    y = normalize_labels(df["Label"])

    X = df.drop("Label", axis=1)

    X = X.select_dtypes(include=[np.number])

    X = X.fillna(0)

    print("Количество объектов:", X.shape[0])
    print("Количество признаков:", X.shape[1])

    return X, y


def train_model(X, y):
    print("Разделение данных...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Обучение модели...")

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    model.fit(X_train, y_train)

    print("Модель обучена")
    print("Проверка модели...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print("\n===== METRICS =====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, predictions, zero_division=0))

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1)
    }

    return model, X_test, y_test, predictions, metrics


def show_class_distribution(y):
    plt.figure(figsize=(10, 5))
    y.value_counts().plot(kind="bar")
    plt.title("Распределение классов")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.show()


def show_confusion_matrix(y_test, predictions):
    labels = sorted(list(pd.Series(y_test).astype(str).unique()))
    cm = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def show_feature_importance(model, X, top_n=20):
    importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n===== TOP IMPORTANT FEATURES =====")
    print(feat_imp.head(top_n))

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feat_imp.head(top_n), x="Importance", y="Feature")
    plt.title(f"Top {top_n} важных признаков")
    plt.tight_layout()
    plt.show()

    feature_path = os.path.join(RESULTS_FOLDER, "feature_importance.csv")
    feat_imp.to_csv(feature_path, index=False)
    print("Важность признаков сохранена:", feature_path)


def save_model_bundle(model, feature_columns, metrics):
    bundle = {
        "model": model,
        "feature_columns": list(feature_columns),
        "binary_mode": BINARY_MODE
    }

    model_path = os.path.join(RESULTS_FOLDER, "ddos_model_bundle.pkl")
    metrics_path = os.path.join(RESULTS_FOLDER, "metrics.json")

    joblib.dump(bundle, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Модель сохранена:", model_path)
    print("Метрики сохранены:", metrics_path)


def predict_file(file_path, model_bundle_path=None):
    if model_bundle_path is None:
        model_bundle_path = os.path.join(RESULTS_FOLDER, "ddos_model_bundle.pkl")

    print("Загрузка модели:", model_bundle_path)
    bundle = joblib.load(model_bundle_path)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    print("Загрузка файла:", file_path)

    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path, low_memory=False)
    else:
        raise Exception("Поддерживаются только файлы .parquet и .csv")

    original_df = df.copy()

    if "Label" in df.columns:
        df = df.drop("Label", axis=1)

    df = df.replace([np.inf, -np.inf], np.nan)

    X = df.select_dtypes(include=[np.number])


    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]
    X = X.fillna(0)

    predictions = model.predict(X)

    result_df = original_df.copy()
    result_df["Prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        class_names = model.classes_

        for i, class_name in enumerate(class_names):
            result_df[f"Prob_{class_name}"] = probabilities[:, i]

    output_path = os.path.join(RESULTS_FOLDER, "predictions_output.csv")
    result_df.to_csv(output_path, index=False)

    print("Предсказания сохранены:", output_path)
    return result_df.head()

def predict_folder(folder_path, model_bundle_path=None):
    if model_bundle_path is None:
        model_bundle_path = os.path.join(RESULTS_FOLDER, "ddos_model_bundle.pkl")

    files = []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.endswith(".parquet") or file.endswith(".csv"):
            if "training" not in file.lower():
                files.append(full_path)

    if len(files) == 0:
        print("Нет файлов для предсказания")
        return None

    print("Найдено файлов для предсказания:", len(files))

    all_results = []

    for file in files:
        print("\nОбработка:", file)

        if file.endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file, low_memory=False)

        original_df = df.copy()

        if "Label" in df.columns:
            df = df.drop("Label", axis=1)

        bundle = joblib.load(model_bundle_path)
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]

        df = df.replace([np.inf, -np.inf], np.nan)
        X = df.select_dtypes(include=[np.number])

        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_columns]
        X = X.fillna(0)

        predictions = model.predict(X)

        original_df["Prediction"] = predictions
        original_df["SourceFile"] = os.path.basename(file)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            class_names = model.classes_
            for i, class_name in enumerate(class_names):
                original_df[f"Prob_{class_name}"] = probabilities[:, i]

        all_results.append(original_df)

    final_df = pd.concat(all_results, ignore_index=True)

    output_path = os.path.join(RESULTS_FOLDER, "all_predictions.csv")
    final_df.to_csv(output_path, index=False)

    print("Все результаты сохранены:", output_path)
    return final_df.head()


def main():
    print("===== DDoS Detector Module =====")

    dataset = load_dataset(DATASET_FOLDER)

    X, y = prepare_data(dataset)

    show_class_distribution(y)

    model, X_test, y_test, predictions, metrics = train_model(X, y)

    show_confusion_matrix(y_test, predictions)
    show_feature_importance(model, X, top_n=20)

    save_model_bundle(model, X.columns, metrics)

    print("\nГотово. Модуль обучен и сохранён.")


#запуск

main()

