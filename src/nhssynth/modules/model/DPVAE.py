from random import gauss

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from pandas import Categorical

# from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from tqdm import tqdm


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
        device="gpu",
    ):
        super().__init__()
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Encoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Encoder: {device} specified, {self.device} used")
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
        device="gpu",
    ):
        super().__init__()

        output_dim = len(singles) + sum([len(x) for x in onehots])
        self.singles = singles
        self.onehots = onehots

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

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

    def __init__(self, encoder, decoder, lr=1e-3):
        super().__init__()
        self.encoder = encoder.to(encoder.device)
        self.decoder = decoder.to(decoder.device)
        self.device = encoder.device
        self.onehots = self.decoder.onehots
        self.singles = self.decoder.singles
        self.noiser = Noiser(len(self.singles)).to(decoder.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr

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
        return x_gen_

    def loss(self, X):
        mu_z, logsigma_z = self.encoder(X)

        p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        q = Normal(mu_z, torch.exp(logsigma_z))

        divergence_loss = torch.sum(torch.distributions.kl_divergence(q, p))

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

        reconstruct_loss = -(categoric_loglik + gauss_loglik)

        elbo = divergence_loss + reconstruct_loss

        return (elbo, reconstruct_loss, divergence_loss, categoric_loglik, gauss_loglik)

    def train(
        self,
        x_dataloader,
        num_epochs,
        logging_freq=1,
        patience=5,
        delta=10,
        filepath=None,
    ):
        # mean_norm = 0
        # counter = 0
        log_elbo = []
        log_reconstruct = []
        log_divergence = []
        log_cat_loss = []
        log_num_loss = []

        # EARLY STOPPING #
        min_elbo = 0.0  # For early stopping workflow
        stop_counter = 0  # Counter for stops

        for epoch in range(num_epochs):

            train_loss = 0.0
            divergence_epoch_loss = 0.0
            reconstruction_epoch_loss = 0.0
            categorical_epoch_reconstruct = 0.0
            numerical_epoch_reconstruct = 0.0

            for batch_idx, (Y_subset,) in enumerate(tqdm(x_dataloader)):
                self.optimizer.zero_grad()
                (
                    elbo,
                    reconstruct_loss,
                    divergence_loss,
                    categorical_reconstruc,
                    numerical_reconstruct,
                ) = self.loss(Y_subset.to(self.encoder.device))
                elbo.backward()
                self.optimizer.step()

                train_loss += elbo.item()
                divergence_epoch_loss += divergence_loss.item()
                reconstruction_epoch_loss += reconstruct_loss.item()
                categorical_epoch_reconstruct += categorical_reconstruc.item()
                numerical_epoch_reconstruct += numerical_reconstruct.item()

                # counter += 1
                # l2_norm = 0
                # for p in self.parameters():
                #     if p.requires_grad:
                #         p_norm = p.grad.detach().data.norm(2)
                #         l2_norm += p_norm.item() ** 2
                # l2_norm = l2_norm ** 0.5  # / Y_subset.shape[0]
                # mean_norm = (mean_norm * (counter - 1) + l2_norm) / counter

            log_elbo.append(train_loss)
            log_reconstruct.append(reconstruction_epoch_loss)
            log_divergence.append(divergence_epoch_loss)
            log_cat_loss.append(categorical_epoch_reconstruct)
            log_num_loss.append(numerical_epoch_reconstruct)

            if epoch == 0:

                min_elbo = train_loss

            if train_loss < (min_elbo - delta):

                min_elbo = train_loss
                stop_counter = 0  # Set counter to zero
                if filepath != None:
                    self.save(filepath)  # Save best model if we want to

            else:  # elbo has not improved

                stop_counter += 1

            if epoch % logging_freq == 0:
                print(
                    f"\tEpoch: {epoch:2}. Elbo: {train_loss:11.2f}. Reconstruction Loss: {reconstruction_epoch_loss:11.2f}. KL Divergence: {divergence_epoch_loss:11.2f}. Categorical Loss: {categorical_epoch_reconstruct:11.2f}. Numerical Loss: {numerical_epoch_reconstruct:11.2f}"
                )
                # print(f"\tMean norm: {mean_norm}")
            # self.mean_norm = mean_norm

            if stop_counter == patience:

                num_epochs = epoch + 1

                break

        return (
            num_epochs,
            log_elbo,
            log_reconstruct,
            log_divergence,
            log_cat_loss,
            log_num_loss,
        )

    def diff_priv_train(
        self,
        x_dataloader,
        num_epochs,
        C=1e16,
        noise_scale=None,
        target_epsilon=1,
        target_delta=1e-5,
        logging_freq=1,
        sample_rate=0.1,
        patience=5,
        delta=10,
        filepath=None,
    ):
        if noise_scale is not None:
            self.privacy_engine = PrivacyEngine(
                self,
                sample_rate=sample_rate,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=noise_scale,
                max_grad_norm=C,
            )
        else:
            self.privacy_engine = PrivacyEngine(
                self,
                sample_rate=sample_rate,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                target_epsilonilon=target_epsilon,
                target_delta=target_delta,
                epochs=num_epochs,
                max_grad_norm=C,
            )
        self.privacy_engine.attach(self.optimizer)

        log_elbo = []
        log_reconstruct = []
        log_divergence = []
        log_cat_loss = []
        log_num_loss = []

        # EARLY STOPPING #
        min_elbo = 0.0  # For early stopping workflow
        patience = patience  # How many epochs patience we give for early stopping
        stop_counter = 0  # Counter for stops
        delta = delta  # Difference in elbo value

        for epoch in range(num_epochs):
            train_loss = 0.0
            divergence_epoch_loss = 0.0
            reconstruction_epoch_loss = 0.0
            categorical_epoch_reconstruct = 0.0
            numerical_epoch_reconstruct = 0.0
            # print(self.get_privacy_spent(target_delta))

            for batch_idx, (Y_subset,) in enumerate(tqdm(x_dataloader)):

                self.optimizer.zero_grad()
                (
                    elbo,
                    reconstruct_loss,
                    divergence_loss,
                    categorical_reconstruct,
                    numerical_reconstruct,
                ) = self.loss(Y_subset.to(self.encoder.device))
                elbo.backward()
                self.optimizer.step()

                train_loss += elbo.item()
                divergence_epoch_loss += divergence_loss.item()
                reconstruction_epoch_loss += reconstruct_loss.item()
                categorical_epoch_reconstruct += categorical_reconstruct.item()
                numerical_epoch_reconstruct += numerical_reconstruct.item()

                # print(self.get_privacy_spent(target_delta))
                # print(loss.item())

            log_elbo.append(train_loss)
            log_reconstruct.append(reconstruction_epoch_loss)
            log_divergence.append(divergence_epoch_loss)
            log_cat_loss.append(categorical_epoch_reconstruct)
            log_num_loss.append(numerical_epoch_reconstruct)

            if epoch == 0:

                min_elbo = train_loss

            if train_loss < (min_elbo - delta):

                min_elbo = train_loss
                stop_counter = 0  # Set counter to zero
                if filepath != None:
                    self.save(filepath)  # Save best model if we want to

            else:  # elbo has not improved

                stop_counter += 1

            if epoch % logging_freq == 0:
                print(
                    f"\tEpoch: {epoch:2}. Elbo: {train_loss:11.2f}. Reconstruction Loss: {reconstruction_epoch_loss:11.2f}. KL Divergence: {divergence_epoch_loss:11.2f}. Categorical Loss: {categorical_epoch_reconstruct:11.2f}. Numerical Loss: {numerical_epoch_reconstruct:11.2f}"
                )
                # print(f"\tMean norm: {mean_norm}")

            if stop_counter == patience:

                num_epochs = epoch + 1
                break

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
