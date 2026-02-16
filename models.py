import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_layers, n_classes):
        super().__init__()
        self.embed = nn.Embedding(n_classes, 8)
        self.lstm = nn.LSTM(z_dim + 8, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, z, c):
        e = self.embed(c).unsqueeze(1).repeat(1, z.size(1), 1)
        x = torch.cat([z, e], dim=-1)
        h, _ = self.lstm(x)
        return self.fc(h)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_classes):
        super().__init__()
        self.embed = nn.Embedding(n_classes, 8)
        self.lstm = nn.LSTM(1 + 8, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, c):
        e = self.embed(c).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, e], dim=-1)
        h, _ = self.lstm(x)
        return self.fc(h[:, -1])
