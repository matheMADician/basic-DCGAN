import torch as t
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utilities.AccuracyChecker as acc
import utilities.DataLoader as load
import utilities.ModelSaver as save

from model import Discriminator, Generator, initialize_weights

device = t.device("cuda" if t.cuda.is_available() else "cpu")

Learning_rate = 2e-4
batch_size = 64
img_size = 64
img_channels = 3
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(img_channels)],
        [0.5 for _ in range(img_channels)]
    )
])

loader = load.Get_loader(
    "custom", batch_size,
    "dataset/", transform= transform
)

gen = Generator(
    z_dim, img_channels, features_gen
).to(device)

disc = Discriminator(
    img_channels, features_disc
).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(
    gen.parameters(), Learning_rate,
    betas= (0.5, 0.999)
)

opt_disc = optim.Adam(
    disc.parameters(), Learning_rate,
    betas= (0.5, 0.999)
)

criterion = nn.BCELoss()

fixed_noise = t.randn(32, z_dim, 1, 1).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        real = real.to(device)
        noise = t.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(real)) + log(1-D(G(z))) 
            # z is random noise
        disc_real = disc(real).reshape(-1)
        lossD_real = criterion(disc_real, t.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)    
        lossD_fake = criterion(disc_fake, t.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()

        ### Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
            # the former one often suffer from gradient vanishing
        output = disc(fake).reshape(-1)
        lossG = criterion(output, t.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == num_epochs-1:
            print(
                f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with t.no_grad():
                fake = gen(fixed_noise)
                
                img_grid_fake = make_grid(fake[:32], normalize=True)
                img_grid_real = make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Fake", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real", img_grid_real, global_step=step
                )
            
            step += 1