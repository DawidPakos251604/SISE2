import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler



def plot_mse_errors(train_error_files, test_error_files):
    plt.figure(figsize=(10, 6))
    for file in train_error_files:
        df = pd.read_csv(file)
        label = os.path.basename(file).replace("errors_", "").replace(".csv", "")
        plt.plot(df["epoch"], df["train_error"], label=label)
    #_, max_val = get_common_ylim(train_error_files, test_error_files)
    #plt.ylim(max_val * 0.5, max_val)  # Obcięcie górnych wartości
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE (zbiór uczący)")
    plt.title("Błąd MSE w trakcie uczenia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mse_train.png")
    plt.close()


def plot_mse_test_errors(train_error_files, test_error_files, raw_error_reference):
    plt.figure(figsize=(10, 6))
    for file in test_error_files:
        df = pd.read_csv(file)
        label = os.path.basename(file).replace("errors_", "").replace(".csv", "")
        plt.plot(df["epoch"], df["test_error"], label=label)
    #_, max_val = get_common_ylim(train_error_files, test_error_files)
    #plt.ylim(max_val * 0.5, max_val)
    plt.axhline(raw_error_reference, color='black', linestyle='--', label="Błąd zmierzony")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE (zbiór testowy)")
    plt.title("Błąd MSE na zbiorze testowym")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mse_test.png")
    plt.close()


def plot_error_cdfs(prediction_files, raw_errors):
    plt.figure(figsize=(10, 6))
    for file in prediction_files:
        df = pd.read_csv(file)
        sorted_errors = np.sort(df["error"])
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        label = os.path.basename(file).replace("predictions_", "").replace(".csv", "")
        plt.plot(sorted_errors, cdf, label=label)

    # Dystrybuanta błędu dla pomiarów dynamicznych
    sorted_raw = np.sort(raw_errors)
    cdf_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
    plt.plot(sorted_raw, cdf_raw, label="pomiar", linestyle='--', color='black')

    plt.xlabel("Błąd [oryginalna skala]")
    plt.ylabel("Dystrybuanta")
    plt.title("Dystrybuanty błędów")
    plt.legend()
    plt.grid(True)
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
    label = os.path.basename(best_file).replace("predictions_", "").replace(".csv", "")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_measured[:, 0], y_measured[:, 1], s=10, c='gray', label="zmierzone")
    plt.scatter(df_best["x"], df_best["y"], s=10, c='orange', label=f"skorygowane ({label})")
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
    train_error_files = sorted(glob.glob("wyniki/errors_*.csv"))
    test_error_files = train_error_files  # Te same pliki zawierają też test_error
    prediction_files = sorted(glob.glob("wyniki/predictions_*.csv"))

    # Odczyt rzeczywistych i zmierzonych danych do scattera i referencji
    y_true = pd.read_csv("wyniki/y_test_true.csv").to_numpy()
    y_measured = pd.read_csv("wyniki/y_test_measured.csv").to_numpy()

    # Skalowanie zmierzonych danych do tej samej skali co dane wyjściowe (przeskalowane wcześniej)
    scaler_y = StandardScaler()
    scaler_y.fit(y_true)  # zakładamy standardową normalizację

    y_measured_scaled = scaler_y.transform(y_measured)
    raw_error_reference = np.mean(np.square(y_measured_scaled - scaler_y.transform(y_true)))

    # 1. Wykres MSE - zbiór uczący
    plot_mse_errors(train_error_files, test_error_files)

    # 2. Wykres MSE - zbiór testowy
    plot_mse_test_errors(train_error_files, test_error_files, raw_error_reference)

    # 3. Dystrybuanty błędów
    all_errors = np.linalg.norm(y_measured - y_true, axis=1)
    plot_error_cdfs(prediction_files, all_errors)

    # 4. Wykres punktowy najlepszego modelu
    plot_best_scatter(prediction_files, y_true, y_measured)

    print(" Wykresy zapisane jako: mse_train.png, mse_test.png, error_cdf.png, scatter_best_model.png")


if __name__ == "__main__":
    main()
