import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ResidualDataset
from models import Discriminator, Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = ResidualDataset("data.npz")
dl = DataLoader(ds, batch_size=128, shuffle=True)

G = Generator(32, 64, 2, 3).to(device)
D = Discriminator(64, 2, 3).to(device)

optG = torch.optim.Adam(G.parameters(), 2e-4)
optD = torch.optim.Adam(D.parameters(), 2e-4)
loss = nn.BCEWithLogitsLoss()

for epoch in range(30):
    for x, c in dl:
        x, c = x.to(device), c.to(device)
        z = torch.randn(x.size(0), x.size(1), 32).to(device)

        fake = G(z, c)
        D_loss = loss(D(x, c), torch.ones(x.size(0), 1).to(device)) + loss(
            D(fake.detach(), c), torch.zeros(x.size(0), 1).to(device)
        )
        optD.zero_grad()
        D_loss.backward()
        optD.step()

        G_loss = loss(D(fake, c), torch.ones(x.size(0), 1).to(device))
        optG.zero_grad()
        G_loss.backward()
        optG.step()

    print(f"Epoch {epoch} | D {D_loss.item():.3f} | G {G_loss.item():.3f}")

torch.save(G.state_dict(), "generator.pt")
print("Saved generator checkpoint to generator.pt")
