import torch
import torchvision.utils
from torch import nn, optim
import numpy as np
from torch.utils.data import dataloader
from torchvision import datasets, transforms

num_epoch = 100
latern_dim = 64
batch_size = 16
img_shape = [1, 28, 28]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, indim):
        super(Generator, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(indim, 128),
            torch.nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(img_shape, dtype=np.int32)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sequential(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(np.prod(img_shape, dtype=np.int32), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sequential(x)
        return out


trans = transforms.Compose(
    {
        transforms.Resize(28),
        transforms.ToTensor()
    }
)

generator = Generator(latern_dim).to(device)
discriminator = Discriminator().to(device)

data_set = datasets.MNIST(root="../data", train=True, transform=trans, download=True)
data_loader = dataloader.DataLoader(dataset=data_set, batch_size=batch_size, drop_last=True)

g_optim = optim.Adam(generator.parameters(), lr=0.0001)
d_optim = optim.Adam(discriminator.parameters(), lr=0.0001)

g_loss = nn.BCELoss().to(device)
d_loss = nn.BCELoss().to(device)

for epoch in range(num_epoch):
    for idx, batch in enumerate(data_loader):
        src_img, tar = batch

        src_img = src_img.to(device)
        # tar = tar.to(device)

        g_optim.zero_grad()
        z = torch.randn(batch_size, latern_dim).to(device)
        pred_img = generator(z)
        g_cost = g_loss(discriminator(pred_img), torch.ones(batch_size, 1).to(device))
        g_cost.backward()
        g_optim.step()

        d_optim.zero_grad()
        src_img = src_img.contiguous().view(batch_size, -1)
        real_loss = d_loss(discriminator(src_img), torch.ones(batch_size, 1).to(device))
        fack_loss = d_loss(discriminator(pred_img.detach()), torch.zeros(batch_size, 1).to(device))
        d_cost = 0.5 * real_loss + 0.5 * fack_loss
        d_cost.backward()
        d_optim.step()
        if idx % 800 == 0:
            print("idx===>", idx, "g_cost===>", g_cost.item(), "d_cost===>", d_cost.item(), "real_loss===>",
                  real_loss.item(),
                  "fake_loss===>", fack_loss.item())
            torchvision.utils.save_image(pred_img.contiguous().view(batch_size, 1, 28, 28),
                                         f"epoch-{epoch}-img-{idx}.png")
