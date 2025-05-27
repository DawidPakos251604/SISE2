import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def extract_activation(file_name):
    """Wydobywa nazwę funkcji aktywacyjnej z nazwy pliku."""
    base = os.path.basename(file_name)
    for act in ["sigmoid", "tanh", "relu"]:
        if act in base:
            return act
    return "unknown"


def get_best_files_per_activation(prediction_files):
    """Zwraca najlepszy plik predykcji (najniższy MSE) dla każdej funkcji aktywacyjnej."""
    best_files = {}
    for file in prediction_files:
        activation = extract_activation(file)
        df = pd.read_csv(file)
        mse = np.mean(df["error"])
        if activation not in best_files or mse < best_files[activation][1]:
            best_files[activation] = (file, mse)
    return [val[0] for val in best_files.values()]


def plot_mse_errors(train_error_files):
    plt.figure(figsize=(10, 6))
    for file in train_error_files:
        df = pd.read_csv(file)
        label = extract_activation(file)
        plt.plot(df["epoch"], df["train_error"], label=label)
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    plt.title("Błąd MSE w trakcie uczenia")
    plt.legend()
    plt.grid(True)
    plt.ylim(top=2)
    plt.tight_layout()
    plt.savefig("mse_train.png")
    plt.close()


def plot_mse_test_errors(test_error_files, raw_error_reference):
    plt.figure(figsize=(10, 6))
    for file in test_error_files:
        df = pd.read_csv(file)
        label = extract_activation(file)
        plt.plot(df["epoch"], df["test_error"], label=label)
    plt.axhline(raw_error_reference, color='black', linestyle='--', label="Błąd zmierzony")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    plt.title("Błąd MSE na zbiorze testowym")
    plt.legend()
    plt.grid(True)
    plt.ylim(top=2)
    plt.tight_layout()
    plt.savefig("mse_test.png")
    plt.close()


def plot_error_cdfs(prediction_files, raw_errors):
    plt.figure(figsize=(10, 6))
    for file in prediction_files:
        df = pd.read_csv(file)
        sorted_errors = np.sort(df["error"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        label = extract_activation(file)
        plt.plot(sorted_errors, cdf, label=label)

    sorted_raw = np.sort(raw_errors)
    cdf_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
    plt.plot(sorted_raw, cdf_raw, label="pomiar", linestyle='--', color='black')

    plt.xlabel("Błąd")
    plt.ylabel("Dystrybuanta")
    plt.title("Dystrybuanty błędów")
    plt.legend()
    plt.grid(True)
    plt.xlim(right=1000)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig("error_cdf.png")
    plt.close()


def plot_best_scatter(prediction_files, y_true, y_measured):
    best_file = None
    best_mse = float('inf')
    for file in prediction_files:
        df = pd.read_csv(file)
        mse = np.mean(df["error"])
        if mse < best_mse:
            best_mse = mse
            best_file = file
    df_best = pd.read_csv(best_file)
    label = extract_activation(file)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_measured[:, 0], y_measured[:, 1], s=10, c='gray', label="zmierzone")
    plt.scatter(df_best["x"], df_best["y"], s=10, c='orange', label=f"{label}")
    plt.scatter(y_true[:, 0], y_true[:, 1], s=10, c='blue', label="rzeczywiste")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Porównanie: rzeczywiste vs. zmierzone vs. skorygowane")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_best_model.png")
    plt.close()


def main():
    all_prediction_files = sorted(glob.glob("wyniki/predictions_*.csv"))
    best_prediction_files = get_best_files_per_activation(all_prediction_files)

    # Odpowiadające pliki błędów
    train_error_files = [
        os.path.join("wyniki", "errors_" + os.path.basename(f).replace("predictions_", ""))
        for f in best_prediction_files
    ]
    test_error_files = train_error_files

    # Odczyt danych rzeczywistych i zmierzonych
    y_true = pd.read_csv("wyniki/y_test_true.csv").to_numpy()
    y_measured = pd.read_csv("wyniki/y_test_measured.csv").to_numpy()

    # Skalowanie do odniesienia błędu
    scaler_y = StandardScaler()
    scaler_y.fit(y_true)

    y_measured_scaled = scaler_y.transform(y_measured)
    raw_error_reference = np.mean(np.square(y_measured_scaled - scaler_y.transform(y_true)))

    # Wykresy
    plot_mse_errors(train_error_files)
    plot_mse_test_errors(test_error_files, raw_error_reference)

    all_errors = np.linalg.norm(y_measured - y_true, axis=1)
    plot_error_cdfs(best_prediction_files, all_errors)

    # Wykres punktowy najlepszego modelu (ze wszystkich)
    plot_best_scatter(all_prediction_files, y_true, y_measured)

    print("Wykresy zapisane jako: mse_train.png, mse_test.png, error_cdf.png, scatter_best_model.png")


if __name__ == "__main__":
    main()
