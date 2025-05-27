import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from model import MLP, ACTIVATIONS

# Ładowanie danych
def load_data(data_dir):
    data = []
    for file in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(file, header=None)
        data.append(df)
    full_data = pd.concat(data, ignore_index=True)
    return full_data

# Przygotowanie danych
def prepare_datasets(train_paths, test_paths):
    def load_multiple(paths):
        dfs = [load_data(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)

    train_data = load_multiple(train_paths)
    test_data = load_multiple(test_paths)

    X_train = train_data.iloc[:, [0, 1]].values
    y_train = train_data.iloc[:, [2, 3]].values
    X_test = test_data.iloc[:, [0, 1]].values
    y_test = test_data.iloc[:, [2, 3]].values
    y_test_measured = test_data.iloc[:, [0, 1]].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

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
        print("Epoka: {epoch}".format(epoch=epoch))
        print("MSE: {loss}".format(loss=train_loss))

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

    print("\n=== Rozpoczynam trening (3 egzemplarze sieci) ===")

    # Dane
    train_dirs = ["dane/f8/stat", "dane/f10/stat"]
    test_dirs = ["dane/f8/dyn", "dane/f10/dyn"]

    X_train, y_train, X_test, y_test_scaled, y_test_true, y_test_measured, scaler_y = prepare_datasets(
        train_dirs, test_dirs
    )

    id_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("wyniki", exist_ok=True)

    # Zapis prawdziwych wartości testowych (w oryginalnej skali)
    pd.DataFrame(y_test_true, columns=["x", "y"]).to_csv("wyniki/y_test_true.csv", index=False)

    # Zapis wartości zmierzonych (w oryginalnej skali)
    pd.DataFrame(y_test_measured, columns=["x", "y"]).to_csv("wyniki/y_test_measured.csv", index=False)

    for i in range(3):
        print(f"\nTrening egzemplarza {i + 1}...")
        model, train_errors, test_errors = train_model(
            activation, hidden, lr, epochs,
            X_train, y_train, X_test, y_test_scaled
        )
        predictions = model(torch.FloatTensor(X_test)).detach().numpy()
        predictions_denorm = scaler_y.inverse_transform(predictions)
        error = np.linalg.norm(predictions_denorm - y_test_true, axis=1)
        key = f"{id_run}_{activation}_{i}"

        # Zapis błędów treningowych i testowych
        df_errors = pd.DataFrame({
            "epoch": list(range(1, epochs + 1)),
            "train_error": train_errors,
            "test_error": test_errors
        })
        df_errors.to_csv(f"wyniki/errors_{key}.csv", index=False)

        # Zapis predykcji i błędów
        df_pred = pd.DataFrame(predictions_denorm, columns=["x", "y"])
        df_pred["error"] = error
        df_pred.to_csv(f"wyniki/predictions_{key}.csv", index=False)

        print(f"Zapisano: wyniki/errors_{key}.csv oraz wyniki/predictions_{key}.csv")

if __name__ == "__main__":
    main()