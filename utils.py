import torch
import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from model import MLP, ACTIVATIONS
from plotting_utils import plot_mse, plot_cdf, scatter_plot


def load_data(paths):
    data = []
    for path in paths:
        for file in glob.glob(os.path.join(path, "*.csv")):
            df = pd.read_csv(file, header=None)
            data.append(df)
    return pd.concat(data, ignore_index=True)


def prepare_datasets(train_paths, test_paths, normalization="minmax"):
    train_data = load_data(train_paths)
    test_data = load_data(test_paths)

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
        scaler_y
    )


def train_model(X_train, y_train, X_test, y_test, hidden_size, activation_fn, learning_rate, epochs):
    model = MLP(hidden_size, activation_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_errors, test_errors = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_output = model(X_test_tensor)
            train_errors.append(criterion(output, y_train_tensor).item())
            test_errors.append(criterion(test_output, y_test_tensor).item())

    return model, train_errors, test_errors


def run_experiment(activation_name, hidden_size, lr, X_train, y_train, X_test, y_test_scaled,
                   y_test_true, y_test_measured, scaler_y, epochs, runs):
    best_model = None
    best_loss = float("inf")
    best_output = None
    errors_all = []
    mse_all = []

    for i in range(runs):
        model, train_errors, test_errors = train_model(
            X_train, y_train, X_test, y_test_scaled, hidden_size,
            ACTIVATIONS[activation_name], lr, epochs
        )

        predictions = model(torch.FloatTensor(X_test)).detach().numpy()
        predictions_denorm = scaler_y.inverse_transform(predictions)
        error = np.linalg.norm(predictions_denorm - y_test_true, axis=1)
        mse = np.mean(error)

        if mse < best_loss:
            best_loss = mse
            best_output = predictions_denorm
            best_model = model

        errors_all.append(error)
        mse_all.append(mse)

    return best_model, best_output, best_loss, errors_all, mse_all


def plot_all_results(all_results, y_test_true, y_test_measured):
    for name, result in all_results.items():
        plot_cdf({name: result["errors"][0]}, np.linalg.norm(y_test_measured - y_test_true, axis=1))
        scatter_plot(y_test_true, y_test_measured, result["best_output"])
        print(f"{name}: MSE = {result['best_loss']:.4f}")
