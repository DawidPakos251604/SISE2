import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_mse(train_errors_dict, test_errors_dict, y_test_scaled, y_test_measured, scaler_y):
    plt.figure(figsize=(10, 5))
    for key, val in train_errors_dict.items():
        plt.plot(val, label=f"Trening - {key}")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title("Błąd MSE na zbiorze treningowym")
    plt.grid(True)
    plt.legend()
    plt.savefig("MSE_train.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for key, val in test_errors_dict.items():
        plt.plot(val, label=f"Test - {key}")
    plt.axhline(y=mean_squared_error(y_test_scaled, scaler_y.transform(y_test_measured)), color='black', linestyle='--', label="Pomiar")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title("Błąd MSE na zbiorze testowym")
    plt.grid(True)
    plt.legend()
    plt.savefig("MSE_test.png")
    plt.close()


def plot_cdf(errors_dict, raw_error):
    plt.figure(figsize=(10, 6))
    for label, errors in errors_dict.items():
        errors_sorted = np.sort(errors)
        cdf = np.linspace(0, 1, len(errors_sorted))
        plt.plot(errors_sorted, cdf, label=label)
    plt.plot(np.sort(raw_error), np.linspace(0, 1, len(raw_error)), linestyle='--', label="Pomiar")
    plt.title("Dystrybuanta błędu (moduł)")
    plt.xlabel("Błąd [m]")
    plt.ylabel("Dystrybuanta")
    plt.grid(True)
    plt.legend()
    plt.savefig("dystrybuanta_bledow.png")
    plt.close()


def scatter_plot(y_true, y_measured, y_corrected):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_measured[:, 0], y_measured[:, 1], alpha=0.3, label="Zmierzone", color="gray")
    plt.scatter(y_corrected[:, 0], y_corrected[:, 1], alpha=0.5, label="Skorygowane", color="blue")
    plt.scatter(y_true[:, 0], y_true[:, 1], alpha=0.9, label="Rzeczywiste", color="red")
    plt.title("Porównanie pozycji")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("scatter_poprawione_vs_rzeczywiste_vs_zmierzony.png")
    plt.close()
