import torch.nn as nn

# Funkcje aktywacji
ACTIVATIONS = { #słownik
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}

class MLP(nn.Module):
    def __init__(self, hidden_size, activation_fn):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_size), 
            activation_fn, #funkcja aktywacji
            nn.Linear(hidden_size, 2)
        )

#Jak dane wejściowe powinny przepływać przez sieć neuronową, do output = model(X_train_tensor)
    def forward(self, x): #metoda propagacji w przód (forward pass)
        return self.model(x)
        #self.model(x) oznacza:
        #Podaj dane x do całej tej sekwencji warstw.
        #PyTorch sam przekaże dane krok po kroku przez wszystkie warstwy.