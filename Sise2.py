import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import cumfreq

from plotting_utils import plot_mse, plot_cdf, scatter_plot


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

def load_data(data_dirs):
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    data = []
    for data_dir in data_dirs:
        for file in glob.glob(os.path.join(data_dir, "*.csv")):
            df = pd.read_csv(file, header=None)
            data.append(df)
    full_data = pd.concat(data, ignore_index=True)
    return full_data



def prepare_datasets(train_paths, test_paths, normalization="minmax"):
    train_data = load_data(train_paths)
    test_data = load_data(test_paths)

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


# Główna pętla
if __name__ == "__main__":
    train_dirs = ["dane/f8/stat", "dane/f10/stat"]
    test_dirs = ["dane/f8/dyn", "dane/f10/dyn"]

    X_train, y_train, X_test, y_test_scaled, y_test_true, y_test_measured, scaler_y = prepare_datasets(
        train_dirs, test_dirs, NORMALIZATION
    )

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
    plot_mse(train_errors_dict, test_errors_dict, y_test_scaled, y_test_measured, scaler_y)
    raw_error = np.linalg.norm(y_test_measured - y_test_true, axis=1)
    plot_cdf(prediction_errors, raw_error)
    scatter_plot(y_test_true, y_test_measured, best_output)

    pd.DataFrame(best_output, columns=["x", "y"]).to_csv("predictions_best.csv", index=False)