import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import cumfreq


# Konfiguracja
HIDDEN_NEURONS = 16
NORMALIZATION = "standard"  # 'minmax', 'maxabs', 'standard'
LEARNING_RATE = 0.001
EPOCHS = 200


# Funkcja aktywacji
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}


# Sieć neuronowa
class MLP(nn.Module):
    def __init__(self, hidden_size, activation_fn):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_size),
            activation_fn,
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.model(x)


# Ładowanie danych

def load_data(data_dir):
    data = []
    for file in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(file, header=None)
        data.append(df)
    full_data = pd.concat(data, ignore_index=True)
    return full_data


def prepare_datasets(train_path, test_path, normalization="minmax"):
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    X_train = train_data.iloc[:, [0, 1]].values
    y_train = train_data.iloc[:, [2, 3]].values
    X_test = test_data.iloc[:, [0, 1]].values
    y_test = test_data.iloc[:, [2, 3]].values
    y_test_measured = test_data.iloc[:, [0, 1]].values  # Zmierzone przed korekcją

    if normalization == "minmax":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif normalization == "maxabs":
        scaler_x = MaxAbsScaler()
        scaler_y = MaxAbsScaler()
    elif normalization == "standard":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    else:
        raise ValueError("Nieznana metoda normalizacji")

    scaler_x.fit(X_train)
    scaler_y.fit(y_train)

    X_train_scaled = scaler_x.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_test, y_test_measured, scaler_y


# Trening

def train_model(activation_name, X_train, y_train, X_test, y_test):
    model = MLP(HIDDEN_NEURONS, ACTIVATIONS[activation_name])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_errors = []
    test_errors = []

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_output = model(X_test_tensor)
            train_loss = criterion(output, y_train_tensor).item()
            test_loss = criterion(test_output, y_test_tensor).item()

        train_errors.append(train_loss)
        test_errors.append(test_loss)

    return model, train_errors, test_errors


# Wykresy błędów

def plot_mse(train_errors_dict, test_errors_dict):
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


# Główna pętla
if __name__ == "__main__":
    train_dir = "dane/f8/stat"
    test_dir = "dane/f8/dyn"

    X_train, y_train, X_test, y_test_scaled, y_test_true, y_test_measured, scaler_y = prepare_datasets(train_dir, test_dir, NORMALIZATION)

    train_errors_dict = {}
    test_errors_dict = {}
    prediction_errors = {}

    best_model = None
    lowest_mse = float("inf")
    best_output = None

    for activation in ACTIVATIONS:
        model, train_errors, test_errors = train_model(activation, X_train, y_train, X_test, y_test_scaled)
        predictions = model(torch.FloatTensor(X_test)).detach().numpy()
        predictions_denorm = scaler_y.inverse_transform(predictions)
        error = np.linalg.norm(predictions_denorm - y_test_true, axis=1)

        train_errors_dict[activation] = train_errors
        test_errors_dict[activation] = test_errors
        prediction_errors[activation] = error

        if np.mean(error) < lowest_mse:
            lowest_mse = np.mean(error)
            best_model = model
            best_output = predictions_denorm

    # Wykresy
    plot_mse(train_errors_dict, test_errors_dict)
    raw_error = np.linalg.norm(y_test_measured - y_test_true, axis=1)
    plot_cdf(prediction_errors, raw_error)
    scatter_plot(y_test_true, y_test_measured, best_output)

    pd.DataFrame(best_output, columns=["x", "y"]).to_csv("predictions_best.csv", index=False)