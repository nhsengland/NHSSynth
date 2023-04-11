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
        use_gpu=False,
    ):
        super().__init__()

        self.device = setup_device(use_gpu)
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
        use_gpu=False,
    ):
        super().__init__()

        self.device = setup_device(use_gpu)
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

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.to(encoder.device)
        self.decoder = decoder.to(decoder.device)
        self.device = encoder.device
        self.onehots = self.decoder.onehots
        self.singles = self.decoder.singles
        self.noiser = Noiser(len(self.singles)).to(decoder.device)

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

        return (elbo, reconstruction_loss, kld, categoric_loglik, gauss_loglik)

    def train(
        self,
        x_dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        privacy_engine: opacus.PrivacyEngine = None,
        patience: int = 5,
        delta: int = 10,
    ):
        if privacy_engine is not None:
            self.privacy_engine = privacy_engine
            self.privacy_engine.attach(self.optimizer)
        # EARLY STOPPING #
        min_elbo = 0.0  # For early stopping workflow
        stop_counter = 0  # Counter for stops

        log_elbo = []
        log_reconstruct = []
        log_divergence = []
        log_cat_loss = []
        log_num_loss = []

        stats_bar_1 = tqdm(total=0, desc="", position=0, bar_format="{desc}", leave=True)
        stats_bar_2 = tqdm(total=0, desc="", position=1, bar_format="{desc}", leave=True)
        stats_bar_3 = tqdm(total=0, desc="", position=2, bar_format="{desc}", leave=True)
        stats_bar_4 = tqdm(total=0, desc="", position=3, bar_format="{desc}", leave=True)
        stats_bar_5 = tqdm(total=0, desc="", position=4, bar_format="{desc}", leave=True)
        position = 5
        if self.privacy_engine is not None:
            epsilon = []
            stats_bar_6 = tqdm(total=0, desc="", position=5, bar_format="{desc}", leave=True)
            position += 1

        for epoch in tqdm(range(num_epochs), desc="Epochs", position=position, leave=False):
            elbo_e = 0.0
            kld_e = 0.0
            reconstruction_e = 0.0
            categorical_e = 0.0
            numerical_e = 0.0

            for (Y_subset,) in tqdm(x_dataloader, desc="Batches", position=position + 1, leave=False):
                self.optimizer.zero_grad()
                (
                    elbo,
                    reconstruction_loss,
                    kld,
                    categorical_loss,
                    numerical_loss,
                ) = self.loss(Y_subset.to(self.encoder.device))
                elbo.backward()
                self.optimizer.step()

                elbo_e += elbo.item()
                kld_e += kld.item()
                reconstruction_e += reconstruction_loss.item()
                categorical_e += categorical_loss.item()
                numerical_e += numerical_loss.item()

            stats_bar_1.set_description_str(f"ELBO: \t\t\t{elbo_e:.2f}")
            stats_bar_2.set_description_str(f"KLD: \t\t\t{kld_e:.2f}")
            stats_bar_3.set_description_str(f"Reconstruction Loss: \t{reconstruction_e:.2f}")
            stats_bar_4.set_description_str(f"Categorical Loss: \t{categorical_e:.2f}")
            stats_bar_5.set_description_str(f"Numerical Loss: \t{numerical_e:.2f}")
            if self.privacy_engine is not None:
                epsilon_e = self.privacy_engine.get_privacy_spent()
                # epsilon_e = self.privacy_engine.accountant.get_epsilon()
                stats_bar_6.set_description_str(f"Epsilon and Best Alpha: \t{epsilon_e[0]:.2f}\t{epsilon_e[1]:.2f}")
                epsilon.append(epsilon_e)

            log_elbo.append(elbo_e)
            log_reconstruct.append(reconstruction_e)
            log_divergence.append(kld_e)
            log_cat_loss.append(categorical_e)
            log_num_loss.append(numerical_e)

            if epoch == 0:
                min_elbo = elbo_e

            if elbo_e < (min_elbo - delta):
                min_elbo = elbo_e
                stop_counter = 0  # Set counter to zero
            else:  # elbo has not improved
                stop_counter += 1

            if stop_counter == patience:
                num_epochs = epoch + 1
                break

        stats_bar_1.close()
        stats_bar_2.close()
        stats_bar_3.close()
        stats_bar_4.close()
        stats_bar_5.close()
        if privacy_engine is not None:
            stats_bar_6.close()
        return (
            num_epochs,
            log_elbo,
            log_reconstruct,
            log_divergence,
            log_cat_loss,
            log_num_loss,
        )

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
