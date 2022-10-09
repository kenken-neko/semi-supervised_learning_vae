from pathlib import Path

import torch
import torch.utils.data
import typer
from ignite.engine import Engine, Events
from ignite.metrics import Loss, MeanSquaredError, RunningAverage
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image

# For MNIST dataset
FLOATTEN_IMAGE_DIMS = 28 * 28


# cf.: https://github.com/pytorch/examples/blob/ca1bd9167f7216e087532160fc5b98643d53f87e/vae/main.py
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(FLOATTEN_IMAGE_DIMS, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, FLOATTEN_IMAGE_DIMS)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # Reparameterization Trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, FLOATTEN_IMAGE_DIMS))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def cal_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(
        recon_x, x.view(-1, FLOATTEN_IMAGE_DIMS), reduction="sum"
    )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld


def log_training(engine, log_name, history, writer):
    avg_loss = engine.state.metrics["loss"]
    print("{}: Epoch {}, Avg loss {}".format(log_name, engine.state.epoch, avg_loss))
    history["loss"].append(engine.state.metrics["loss"])
    writer.add_scalar(f"{log_name}/loss", avg_loss, engine.state.epoch)


def log_validation(engine, evaluator, dataloader, log_name, history, writer):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_loss = metrics["loss"]
    avg_mse = metrics["mse"]
    print(
        "{}: Epoch {}, Avg loss {}, Avg mse {}".format(
            log_name, engine.state.epoch, avg_loss, avg_mse
        )
    )
    for key in evaluator.state.metrics.keys():
        history[key].append(evaluator.state.metrics[key])
    writer.add_scalar(f"{log_name}/loss", avg_loss, engine.state.epoch)
    writer.add_scalar(f"{log_name}/mse", avg_mse, engine.state.epoch)


def main(
    num_workers: int = typer.Option(
        1, help="Number of workers for train and validation data loader"
    ),
    batch_size: int = typer.Option(
        32, help="Batch size for train and validation data loader"
    ),
    max_epochs: int = typer.Option(20, help="Maximum epoch number"),
    log_dir: Path = typer.Option("logs", help="Log directory"),
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, _ = batch
        x = x.to(device)
        x = x.view(-1, FLOATTEN_IMAGE_DIMS)
        x_pred, mu, logvar = model(x)
        loss = cal_loss(x_pred, x, mu, logvar)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, _ = batch
            x = x.to(device)
            x = x.view(-1, FLOATTEN_IMAGE_DIMS)
            x_pred, mu, logvar = model(x)
            kwargs = {"mu": mu, "logvar": logvar}
            return x_pred, x, kwargs

    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)
    training_history = {'loss': []}
    validation_history = {"loss": [], "mse": []}

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    # Egnite `Loss` is specified by three variables `y_pred`, `y`, and `kwargs`
    Loss(cal_loss, output_transform=lambda x: [x[0], x[1], x[2]]).attach(
        evaluator, "loss"
    )
    MeanSquaredError(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, "mse")

    data_transform = Compose([ToTensor()])
    train_data = MNIST(
        download=True, root="/tmp/mnist/", transform=data_transform, train=True
    )
    val_data = MNIST(
        download=True, root="/tmp/mnist/", transform=data_transform, train=False
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    writer_training = SummaryWriter(logdir=log_dir / "training")
    writer_validation = SummaryWriter(logdir=log_dir / "validation")
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_training,
        "Training_and_Validation",
        training_history,
        writer_training,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_validation,
        evaluator,
        val_loader,
        "Training_and_Validation",
        validation_history,
        writer_validation,
    )
    writer_training.close()
    writer_validation.close()

    trainer.run(train_loader, max_epochs=max_epochs)


if __name__ == "__main__":
    typer.run(main)
