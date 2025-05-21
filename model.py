import torch.nn as nn

# Funkcje aktywacji
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}

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