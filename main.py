import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error

from plotting_utils import plot_mse, plot_cdf, scatter_plot

# Funkcje aktywacji
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}

# Model
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

# Przygotowanie danych
def prepare_datasets(train_path, test_path, normalization="minmax"):
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    X_train = train_data.iloc[:, [0, 1]].values
    y_train = train_data.iloc[:, [2, 3]].values
    X_test = test_data.iloc[:, [0, 1]].values
    y_test = test_data.iloc[:, [2, 3]].values
    y_test_measured = test_data.iloc[:, [0, 1]].values

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

    return (
        scaler_x.transform(X_train),
        scaler_y.transform(y_train),
        scaler_x.transform(X_test),
        scaler_y.transform(y_test),
        y_test,
        y_test_measured,
        scaler_y,
    )

# Trenowanie modelu
def train_model(activation_name, hidden_neurons, learning_rate, epochs, X_train, y_train, X_test, y_test):
    model = MLP(hidden_neurons, ACTIVATIONS[activation_name])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_errors = []
    test_errors = []

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    for epoch in range(epochs):
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

# Główna funkcja
def main():
    print("=== MENU KONFIGURACJI SIECI MLP ===")

    # Funkcja aktywacji
    while True:
        activation = input("Wybierz funkcję aktywacji (relu/sigmoid/tanh): ").strip().lower()
        if activation in ACTIVATIONS:
            break
        print("Niepoprawna funkcja aktywacji. Spróbuj ponownie.")

    # Liczba neuronów
    while True:
        try:
            hidden = int(input("Podaj liczbę neuronów w warstwie ukrytej: "))
            if hidden > 0:
                break
        except ValueError:
            pass
        print("Nieprawidłowa liczba neuronów.")

    # Learning rate
    while True:
        try:
            lr = float(input("Podaj wartość learning rate (np. 0.001): "))
            if lr > 0:
                break
        except ValueError:
            pass
        print("Nieprawidłowa wartość learning rate.")

    # Epoki
    while True:
        try:
            epochs = int(input("Podaj liczbę epok: "))
            if epochs > 0:
                break
        except ValueError:
            pass
        print("Nieprawidłowa liczba epok.")

    # Normalizacja
    while True:
        normalization = input("Rodzaj normalizacji (minmax/maxabs/standard) [domyślnie: standard]: ").strip().lower()
        if normalization == "":
            normalization = "standard"
        if normalization in ["minmax", "maxabs", "standard"]:
            break
        print("Nieprawidłowy wybór normalizacji.")

    print("\n=== Rozpoczynam trening (3 egzemplarze sieci) ===")

    # Dane
    train_dir = "dane/f8/stat"
    test_dir = "dane/f8/dyn"

    X_train, y_train, X_test, y_test_scaled, y_test_true, y_test_measured, scaler_y = prepare_datasets(
        train_dir, test_dir, normalization
    )

    best_model = None
    lowest_mse = float("inf")
    best_output = None
    train_errors_dict = {}
    test_errors_dict = {}
    prediction_errors = {}

    for i in range(3):
        print(f"\nTrening egzemplarza {i + 1}...")
        model, train_errors, test_errors = train_model(
            activation, hidden, lr, epochs,
            X_train, y_train, X_test, y_test_scaled
        )
        predictions = model(torch.FloatTensor(X_test)).detach().numpy()
        predictions_denorm = scaler_y.inverse_transform(predictions)
        error = np.linalg.norm(predictions_denorm - y_test_true, axis=1)

        key = f"{activation}_{i}"
        train_errors_dict[key] = train_errors
        test_errors_dict[key] = test_errors
        prediction_errors[key] = error

        mse = np.mean(error)
        print(f"Egzemplarz {i + 1}: MSE = {mse:.4f}")

        if mse < lowest_mse:
            lowest_mse = mse
            best_model = model
            best_output = predictions_denorm

    print(f"\n✅ Najlepszy wynik MSE: {lowest_mse:.4f}")
    print("Tworzę wykresy i zapisuję dane...")

    raw_error = np.linalg.norm(y_test_measured - y_test_true, axis=1)
    plot_mse(train_errors_dict, test_errors_dict)
    plot_cdf(prediction_errors, raw_error)
    scatter_plot(y_test_true, y_test_measured, best_output)

    out_filename = f"best_prediction_{activation}.csv"
    pd.DataFrame(best_output, columns=["x", "y"]).to_csv(out_filename, index=False)
    print(f"Wynik zapisany do pliku: {out_filename}")


if __name__ == "__main__":
    main()
