import matplotlib.pyplot as plt
import numpy as np


def plot_mse(train_errors_dict, test_errors_dict):
    """
    Wykresy MSE dla zbiorów uczącego i testowego dla różnych funkcji aktywacji.
    """
    plt.figure(figsize=(12, 6))

    for name, errors in train_errors_dict.items():
        plt.plot(errors, label=f'{name} - train', linestyle='--')

    for name, errors in test_errors_dict.items():
        plt.plot(errors, label=f'{name} - test')

    plt.title("MSE w trakcie epok")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cdf(prediction_errors_dict, raw_error):
    """
    Wykres CDF (empiryczna dystrybuanta) dla błędów predykcji.
    """
    plt.figure(figsize=(10, 6))

    for name, errors in prediction_errors_dict.items():
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
        plt.plot(sorted_errors, cdf, label=f'{name} - corrected')

    sorted_raw = np.sort(raw_error)
    cdf_raw = np.arange(1, len(sorted_raw)+1) / len(sorted_raw)
    plt.plot(sorted_raw, cdf_raw, label='measured (raw)', color='black', linestyle='--')

    plt.title("CDF błędu lokalizacji")
    plt.xlabel("Błąd [jednostki]")
    plt.ylabel("Prawdopodobieństwo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def scatter_plot(y_true, y_measured, y_corrected):
    """
    Wykres rozrzutu: porównanie współrzędnych rzeczywistych, zmierzonych i skorygowanych.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true[:, 0], y_true[:, 1], label='Rzeczywiste', c='green', s=50, alpha=0.6)
    plt.scatter(y_measured[:, 0], y_measured[:, 1], label='Zmierzone', c='red', marker='x')
    plt.scatter(y_corrected[:, 0], y_corrected[:, 1], label='Skorygowane', c='blue', marker='o', edgecolors='k')

    plt.title("Porównanie lokalizacji")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
