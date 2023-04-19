import time
import warnings

import opacus
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm


def setup_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            warnings.warn("`use_gpu` was provided but no GPU is available, using CPU")
    return torch.device("cpu")


class Encoder(nn.Module):
    """Encoder, takes in x
    and outputs mu_z, sigma_z
    (diagonal Gaussian variational posterior assumed)
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim=32,
        activation=nn.Tanh,
    ):
        super().__init__()

        output_dim = 2 * latent_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        outs = self.net(x)
        mu_z = outs[:, : self.latent_dim]
        logsigma_z = outs[:, self.latent_dim :]
        return mu_z, logsigma_z


class Decoder(nn.Module):
    """Decoder, takes in z and outputs reconstruction"""

    def __init__(
        self,
        latent_dim,
        onehots=[[]],
        singles=[],
        hidden_dim=32,
        activation=nn.Tanh,
    ):
        super().__init__()

        output_dim = len(singles) + sum([len(x) for x in onehots])
        self.singles = singles
        self.onehots = onehots

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Noiser(nn.Module):
    def __init__(self, num_singles):
        super().__init__()
        self.output_logsigma_fn = nn.Linear(num_singles, num_singles, bias=True)
        torch.nn.init.zeros_(self.output_logsigma_fn.weight)
        torch.nn.init.zeros_(self.output_logsigma_fn.bias)
        self.output_logsigma_fn.weight.requires_grad = False

    def forward(self, X):
        return self.output_logsigma_fn(X)


class VAE(nn.Module):
    """Combines encoder and decoder into full VAE model"""

    def __init__(self, encoder, decoder, e_optimizer, d_optimizer, use_gpu=False):
        super().__init__()
        self.device = setup_device(use_gpu)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.e_optimizer = e_optimizer
        self.d_optimizer = d_optimizer
        self.onehots = self.decoder.onehots
        self.singles = self.decoder.singles
        self.noiser = Noiser(len(self.singles)).to(self.device)

    def reconstruct(self, X):
        mu_z, logsigma_z = self.encoder(X)

        x_recon = self.decoder(mu_z)
        return x_recon

    def generate(self, N):
        z_samples = torch.randn_like(torch.ones((N, self.encoder.latent_dim)), device=self.device)
        x_gen = self.decoder(z_samples)
        x_gen_ = torch.ones_like(x_gen, device=self.device)

        for cat_idxs in self.onehots:
            x_gen_[:, cat_idxs] = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=x_gen[:, cat_idxs]
            ).sample()

        x_gen_[:, self.singles] = x_gen[:, self.singles] + torch.exp(
            self.noiser(x_gen[:, self.singles])
        ) * torch.randn_like(x_gen[:, self.singles])
        if torch.cuda.is_available():
            x_gen_ = x_gen_.cpu()
        return x_gen_.detach()

    def loss(self, X):
        mu_z, logsigma_z = self.encoder(X)

        p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        q = Normal(mu_z, torch.exp(logsigma_z))

        kld = torch.sum(torch.distributions.kl_divergence(q, p))

        s = torch.randn_like(mu_z)
        z_samples = mu_z + s * torch.exp(logsigma_z)

        x_recon = self.decoder(z_samples)

        categoric_loglik = 0
        if len(self.onehots):
            for cat_idxs in self.onehots:
                categoric_loglik += -torch.nn.functional.cross_entropy(
                    x_recon[:, cat_idxs],
                    torch.max(X[:, cat_idxs], 1)[1],
                ).sum()

        gauss_loglik = 0
        if len(self.singles):
            gauss_loglik = (
                Normal(
                    loc=x_recon[:, self.singles],
                    scale=torch.exp(self.noiser(x_recon[:, self.singles])),
                )
                .log_prob(X[:, self.singles])
                .sum()
            )

        reconstruction_loss = -(categoric_loglik + gauss_loglik)

        elbo = kld + reconstruction_loss

        return {
            "ELBO": elbo,
            "ReconstructionLoss": reconstruction_loss,
            "KLD": kld,
            "CategoricalLoss": categoric_loglik,
            "NumericalLoss": gauss_loglik,
        }

    def train(
        self,
        x_dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        tracked_metrics: list[str] = ["ELBO"],
        privacy_engine: opacus.PrivacyEngine = None,
        patience: int = 5,
        delta: int = 10,
    ):
        print("")

        self.start_time = time.time()

        if privacy_engine is not None:
            self.privacy_engine = privacy_engine
        elif "Privacy \u03B5" in tracked_metrics:
            tracked_metrics.remove("Privacy \u03B5")

        min_elbo = 0.0  # For early stopping workflow
        stop_counter = 0  # Counter for stops

        metrics = {metric: [] for metric in tracked_metrics}
        stats_bars = {
            metric: tqdm(total=0, desc="", position=i, bar_format="{desc}", leave=True)
            for i, metric in enumerate(tracked_metrics)
        }
        max_length = max(len(s) for s in tracked_metrics) + 1

        for epoch in tqdm(range(num_epochs), desc="Epochs", position=len(stats_bars), leave=False):
            for key in metrics.keys():
                if key != "Privacy \u03B5":
                    metrics[key].append(0.0)

            for (Y_subset,) in tqdm(x_dataloader, desc="Batches", position=len(stats_bars) + 1, leave=False):
                self.e_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                losses = self.loss(Y_subset.to(self.encoder.device))
                losses["ELBO"].backward()
                self.e_optimizer.step()
                self.d_optimizer.step()

                for key in metrics.keys():
                    if key in losses:
                        if losses[key]:
                            metrics[key][-1] += losses[key].item()
                        else:
                            metrics[key][-1] += 0.0

            for key, stats_bar in stats_bars.items():
                if key == "Privacy \u03B5" and privacy_engine is not None:
                    epsilon_e = self.privacy_engine.accountant.get_epsilon()
                    metrics[key][-1] += epsilon_e
                stats_bar.set_description_str(f"{(key + ':').ljust(max_length)}  {metrics[key][-1]:.2f}")

            if epoch == 0:
                min_elbo = metrics["ELBO"][-1]

            if metrics["ELBO"][-1] < (min_elbo - delta):
                min_elbo = metrics["ELBO"][-1]
                stop_counter = 0  # Set counter to zero
            else:  # elbo has not improved
                stop_counter += 1

            if stop_counter == patience:
                num_epochs = epoch + 1
                break

        for stats_bar in stats_bars.values():
            stats_bar.close()

        tqdm.write(f"Completed {num_epochs} epochs in {time.time() - self.start_time:.2f} seconds.")

        return (num_epochs, metrics)

    def get_privacy_spent(self, delta):
        if hasattr(self, "privacy_engine"):
            return self.privacy_engine.get_privacy_spent(delta)
        else:
            print(
                """This VAE object does not a privacy_engine attribute.
                Run diff_priv_train to create one."""
            )

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
