import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def extract_activation(file_name):
    base = os.path.basename(file_name)
    for act in ["sigmoid", "tanh", "relu"]:
        if act in base:
            return act
    return "unknown"


def get_best_files_by_test_error_per_activation(error_files_dir):
    best_files = {}
    error_files = sorted(glob.glob(os.path.join(error_files_dir, "errors_*.csv")))

    for file in error_files:
        activation = extract_activation(file)
        df = pd.read_csv(file)
        last_test_error = df["test_error"].iloc[-1]
        if activation not in best_files or last_test_error < best_files[activation][1]:
            best_files[activation] = (file, last_test_error)

    best_error_files = [v[0] for v in best_files.values()]
    best_prediction_files = [
        os.path.join(
            error_files_dir,
            "predictions_" + os.path.basename(f).replace("errors_", "")
        ) for f in best_error_files
    ]

    # Najlepszy ogólnie
    best_overall = min(best_files.items(), key=lambda x: x[1][1])
    best_activation = best_overall[0]
    best_error_file = best_overall[1][0]
    best_prediction_file = os.path.join(
        error_files_dir,
        "predictions_" + os.path.basename(best_error_file).replace("errors_", "")
    )

    return best_prediction_files, best_error_files, best_prediction_file, best_activation, best_files


def plot_mse_errors(train_error_files):
    plt.figure(figsize=(10, 6))
    for file in train_error_files:
        df = pd.read_csv(file)
        label = extract_activation(file)
        plt.plot(df["epoch"], df["train_error"], label=label)
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    plt.title("Błąd MSE na zbiorze treningowym")
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
    plt.xlim(right=800)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig("error_cdf.png")
    plt.close()


def plot_best_scatter(prediction_file, y_true, y_measured, label):
    df_best = pd.read_csv(prediction_file)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_measured[:, 0], y_measured[:, 1], s=10, c='#ead36e', label="zmierzone", zorder=1)
    plt.scatter(df_best["x"], df_best["y"], s=10, c='#57c547', label=f"skorygowane ({label})", zorder=2)
    plt.scatter(y_true[:, 0], y_true[:, 1], s=10, c='#005589', label="rzeczywiste", zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_best_model.png")
    plt.close()


def main():
    best_prediction_files, best_error_files, best_prediction_file, best_activation, best_files = \
        get_best_files_by_test_error_per_activation("wyniki")

    y_true = pd.read_csv("wyniki/y_test_true.csv").to_numpy()
    y_measured = pd.read_csv("wyniki/y_test_measured.csv").to_numpy()

    scaler_y = StandardScaler()
    scaler_y.fit(y_true)
    y_measured_scaled = scaler_y.transform(y_measured)
    raw_error_reference = np.mean(np.square(y_measured_scaled - scaler_y.transform(y_true)))

    plot_mse_errors(best_error_files)
    plot_mse_test_errors(best_error_files, raw_error_reference)

    all_errors = np.linalg.norm(y_measured - y_true, axis=1)
    plot_error_cdfs(best_prediction_files, all_errors)

    plot_best_scatter(best_prediction_file, y_true, y_measured, best_activation)

    print("Wykresy zapisane jako: mse_train.png, mse_test.png, error_cdf.png, scatter_best_model.png\n")

    print("Najlepsze modele dla każdej funkcji aktywacyjnej:")
    for activation, (file_path, test_error) in best_files.items():
        print(f"- {activation}: {os.path.basename(file_path)} (test_error = {test_error:.4f})")

    print(f"\n Najlepszy ogólny model: {os.path.basename(best_prediction_file)} (aktywacja: {best_activation})")


if __name__ == "__main__":
    main()
