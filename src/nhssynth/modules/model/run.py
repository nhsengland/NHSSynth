import argparse

import pandas as pd
import torch
from nhssynth.common import *
from nhssynth.modules.model.DPVAE import VAE, Decoder, Encoder
from nhssynth.modules.model.io import check_output_paths, load_required_data
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader, TensorDataset


def run(args: argparse.Namespace) -> argparse.Namespace:
    """Run the model architecture module."""
    print("Running model architecture module...")

    set_seed(args.seed)

    dir_experiment = experiment_io(args.experiment_name)

    fn_base, data, mt = load_required_data(args, dir_experiment)
    data, categoricals, num_continuous = mt.order(data)
    nrows, ncols = data.shape

    torch_data = TensorDataset(torch.Tensor(data.to_numpy()))
    sample_rate = args.batch_size / nrows
    data_loader = DataLoader(
        torch_data,
        batch_sampler=UniformWithReplacementSampler(num_samples=nrows, sample_rate=sample_rate),
        pin_memory=True,
    )

    print(f"Train, generate and evaluate {'' if args.non_private_training else 'DP'}VAE...")

    encoder = Encoder(ncols, args.latent_dim, hidden_dim=args.hidden_dim)
    decoder = Decoder(args.latent_dim, num_continuous, num_categories=categoricals, hidden_dim=args.hidden_dim)
    vae = VAE(encoder, decoder)
    if not args.non_private_training:
        results = vae.diff_priv_train(
            data_loader,
            num_epochs=args.num_epochs,
            C=args.max_grad_norm,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(args.target_delta)}")
    else:
        results = vae.train(data_loader, num_epochs=args.num_epochs)

    synthetic_data = vae.generate(nrows)

    if torch.cuda.is_available():
        synthetic_data = synthetic_data.cpu()

    synthetic_data = pd.DataFrame(synthetic_data.detach(), columns=data.columns)
    fn_output, fn_model = check_output_paths(fn_base, args.synthetic_data, args.model_file, dir_experiment)
    if not args.discard_data:
        synthetic_data = mt.inverse_apply(synthetic_data)
        synthetic_data.to_csv(dir_experiment / fn_output, index=False)
    if not args.discard_model:
        vae.save(dir_experiment / fn_model)

    if args.modules_to_run and "evaluation" in args.modules_to_run:
        args.model_output = {"results": results}

    return args
