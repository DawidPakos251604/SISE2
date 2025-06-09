import os  # Operacje systemowe (np. tworzenie folderów, ścieżki plików)
import glob  # Wyszukiwanie plików po wzorcach (np. *.csv)
import torch  # Główna biblioteka PyTorch
import torch.nn as nn  # Moduł do warstw sieciowych i funkcji kosztu
import torch.optim as optim  # Optymalizatory (np. Adam)
import pandas as pd  # Obsługa danych tabelarycznych (DataFrame)
import numpy as np  # Obsługa macierzy i obliczeń matematycznych
from sklearn.preprocessing import StandardScaler  # Skalowanie danych (normalizacja, standaryzacja)
from datetime import datetime  # Czas i daty, np. do oznaczania zapisanych plików

from model import MLP, ACTIVATIONS  # Własny moduł z definicją modelu i słownikiem aktywacji

# Ładowanie danych - Wczytuje wszystkie pliki .csv z danego folderu data_dir i łączy je w jeden duży DataFrame.
def load_data(data_dir):
    data = []  # Lista na wszystkie DataFrame'y
    for file in glob.glob(os.path.join(data_dir, "*.csv")):  # Przeszukuje folder pod kątem plików CSV
        df = pd.read_csv(file, header=None)  # Wczytuje CSV bez nagłówka jako DataFrame
        data.append(df)  # Dodaje do listy
    full_data = pd.concat(data, ignore_index=True)  # Łączy wszystkie DataFrame'y w jeden duży
    return full_data

# Przygotowanie danych treningowych i testowych
def prepare_datasets(train_paths, test_paths):
    def load_multiple(paths):
        dfs = [load_data(p) for p in paths]  # Dla każdej ścieżki ładuje dane
        return pd.concat(dfs, ignore_index=True)  # Łączy dane w jeden DataFrame

    train_data = load_multiple(train_paths)  # Ładowanie danych treningowych
    test_data = load_multiple(test_paths)  # Ładowanie danych testowych

    # Wydzielenie cech wejściowych (kolumny 0 i 1) i wyjściowych (kolumny 2 i 3)
    X_train = train_data.iloc[:, [0, 1]].values
    y_train = train_data.iloc[:, [2, 3]].values
    X_test = test_data.iloc[:, [0, 1]].values
    y_test = test_data.iloc[:, [2, 3]].values
    y_test_measured = test_data.iloc[:, [0, 1]].values  # Zmierzony punkt wejściowy, nieprzeskalowany

    # Skalery do normalizacji danych (tylko na danych treningowych!)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y.fit(y_train)

    return (
        scaler_x.transform(X_train),  # Wejścia treningowe - przeskalowane
        scaler_y.transform(y_train),  # Wyjścia treningowe - przeskalowane
        scaler_x.transform(X_test),   # Wejścia testowe - przeskalowane
        scaler_y.transform(y_test),   # Wyjścia testowe - przeskalowane
        y_test,                       # Wyjścia testowe (oryginalne) - do późniejszej denormalizacji
        y_test_measured,             # Wejścia testowe (oryginalne)
        scaler_y                     # Skaler Y do odwrócenia skalowania wyników
    )

# Funkcja ucząca model
def train_model(activation_name, hidden_neurons, learning_rate, epochs, X_train, y_train, X_test, y_test):
    model = MLP(hidden_neurons, ACTIVATIONS[activation_name])  # Tworzy sieć MLP z wybraną funkcją aktywacji

    # Funkcja aktywacji to nieliniowa funkcja stosowana na wyjściu neuronu, która decyduje o tym, czy i w jakim stopniu sygnał zostanie przekazany dalej.
    # Dzięki niej sieć może uczyć się złożonych, nieliniowych zależności między danymi.
    # Gdyby nie zastosować funkcji aktywacji (lub gdyby była liniowa),
    # cała sieć byłaby tylko kombinacją liniową, co ograniczałoby jej zdolność do modelowania skomplikowanych wzorców.

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Ustawienie optymalizatora
    criterion = nn.MSELoss()  # Funkcja kosztu - średni błąd kwadratowy

    train_errors = []  # Lista błędów treningowych
    test_errors = []   # Lista błędów testowych

    # Zamiana danych z NumPy na tensory PyTorch
    # PyTorch działa tylko na Tensorach (swoim własnym typie danych)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    for epoch in range(epochs):  # Pętla po epokach
        model.train()  # Ustawienie modelu w tryb treningowy
        optimizer.zero_grad()  # Wyzerowanie gradientów z poprzedniej epoki

        output = model(X_train_tensor)  # Predykcja na danych treningowych
        #output = activation(weights * input + bias)
        #Propagacja w przód (forward pass): dane wejściowe przechodzą przez warstwy sieci
        # (macierzowe mnożenia, dodanie biasów, aktywacja).
        # Na końcu sieć zwraca wektor wyjściowy – predykcję modelu.

        loss = criterion(output, y_train_tensor)  # Obliczenie straty (błędu)
        #porównujemy predykcję output z prawdziwymi wartościami y_train_tensor za pomocą funkcji strat

        #tzn. jak bardzo każda waga przyczyniła się do błędu
        loss.backward()  # Obliczenie gradientów (wsteczna propagacja)

        #PyTorch używa wcześniej policzonych gradientów i aktualizuje wagi w sieci
        optimizer.step()  # Aktualizacja wag - zmienia je tak, by w kolejnej epoce błąd był mniejszy

        model.eval()  # Przełączenie modelu w tryb ewaluacyjny
        with torch.no_grad():  # Bez liczenia gradientów - szybciej
            # Nie zużywamy pamięci na zapisywanie operacji potrzebnych do backward(),
            # Predykcja działa szybciej
            # Nie ryzykujemy przypadkowego nadpisania gradientów.
            test_output = model(X_test_tensor)  # Predykcja na danych testowych
            train_loss = criterion(output, y_train_tensor).item()  # Błąd treningowy
            test_loss = criterion(test_output, y_test_tensor).item()  # Błąd testowy

        train_errors.append(train_loss)
        test_errors.append(test_loss)
        print("Epoka: {epoch}".format(epoch=epoch))
        print("MSE: {loss}".format(loss=train_loss))  # Tylko błąd treningowy wypisywany

    return model, train_errors, test_errors  # Zwraca wytrenowany model i listy błędów

# Główna funkcja uruchomieniowa programu
def main():
    print("=== MENU KONFIGURACJI SIECI MLP ===")

    # Interfejs tekstowy do wyboru funkcji aktywacji
    while True:
        activation = input("Wybierz funkcję aktywacji (relu/sigmoid/tanh): ").strip().lower()
        if activation in ACTIVATIONS:
            break
        print("Niepoprawna funkcja aktywacji. Spróbuj ponownie.")

    # Podanie liczby neuronów w warstwie ukrytej
    while True:
        try:
            hidden = int(input("Podaj liczbę neuronów w warstwie ukrytej: "))
            if hidden > 0:
                break
        except ValueError:
            pass
        print("Nieprawidłowa liczba neuronów.")

    # Podanie learning rate
    while True:
        try:
            lr = float(input("Podaj wartość learning rate (np. 0.001): "))
            if lr > 0:
                break
        except ValueError:
            pass
        print("Nieprawidłowa wartość learning rate.")

    # Podanie liczby epok
    while True:
        try:
            epochs = int(input("Podaj liczbę epok: "))
            if epochs > 0:
                break
        except ValueError:
            pass
        print("Nieprawidłowa liczba epok.")

    print("\n=== Rozpoczynam trening (3 egzemplarze sieci) ===")

    # Wskazanie folderów z danymi
    train_dirs = ["dane/f8/stat", "dane/f10/stat"]  # statyczne dane do treningu
    test_dirs = ["dane/f8/dyn", "dane/f10/dyn"]     # dynamiczne dane do testowania

    # Przygotowanie danych (przeskalowane i oryginalne)
    X_train, y_train, X_test, y_test_scaled, y_test_true, y_test_measured, scaler_y = prepare_datasets(
        train_dirs, test_dirs
    )

    id_run = datetime.now().strftime("%Y%m%d_%H%M%S")  # Znacznik czasowy dla zapisów
    os.makedirs("wyniki", exist_ok=True)  # Tworzy folder wyniki, jeśli nie istnieje

    # Zapis danych testowych (oryginalnych)
    pd.DataFrame(y_test_true, columns=["x", "y"]).to_csv("wyniki/y_test_true.csv", index=False)
    pd.DataFrame(y_test_measured, columns=["x", "y"]).to_csv("wyniki/y_test_measured.csv", index=False)

    # Trenuje 3 egzemplarze modelu z tymi samymi danymi i parametrami
    for i in range(3):
        print(f"\nTrening egzemplarza {i + 1}...")
        model, train_errors, test_errors = train_model(
            activation, hidden, lr, epochs,
            X_train, y_train, X_test, y_test_scaled
        )

        # Predykcja na danych testowych i denormalizacja wyników
        predictions = model(torch.FloatTensor(X_test)).detach().numpy()  # Predykcja (skala znormalizowana)
        predictions_denorm = scaler_y.inverse_transform(predictions)  # Odwrócenie skalowania
        error = np.linalg.norm(predictions_denorm - y_test_true, axis=1)  # Obliczenie błędu (euklidesowego)

        key = f"{id_run}_{activation}_{i}"  # Nazwa pliku na podstawie czasu, funkcji aktywacji i indeksu

        # Zapis błędów treningowych i testowych do pliku
        df_errors = pd.DataFrame({
            "epoch": list(range(1, epochs + 1)),
            "train_error": train_errors,
            "test_error": test_errors
        })
        df_errors.to_csv(f"wyniki/errors_{key}.csv", index=False)

        # Zapis predykcji i błędów do pliku
        df_pred = pd.DataFrame(predictions_denorm, columns=["x", "y"])
        df_pred["error"] = error
        df_pred.to_csv(f"wyniki/predictions_{key}.csv", index=False)

        print(f"Zapisano: wyniki/errors_{key}.csv oraz wyniki/predictions_{key}.csv")

if __name__ == "__main__":
    main()
