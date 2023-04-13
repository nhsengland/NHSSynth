import argparse

import pandas as pd
import torch
from nhssynth.common import *
from nhssynth.modules.model.DPVAE import VAE, Decoder, Encoder
from nhssynth.modules.model.io import check_output_paths, load_required_data
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader, TensorDataset


# TODO clean this up when we move beyond the old DPVAE
def run(args: argparse.Namespace) -> argparse.Namespace:
    """Run the model architecture module."""
    print("Running model architecture module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, prepared_dataset, mt = load_required_data(args, dir_experiment)
    onehots, singles = mt.get_onehots_and_singles()
    nrows, ncols = prepared_dataset.shape

    # Should the data also all be turned into floats?
    torch_data = TensorDataset(torch.Tensor(prepared_dataset.to_numpy()))
    sample_rate = args.batch_size / nrows
    model = VAE(
        Encoder(input_dim=ncols, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, use_gpu=args.use_gpu),
        Decoder(args.latent_dim, onehots=onehots, singles=singles, use_gpu=args.use_gpu),
    )
    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    data_loader = DataLoader(
        torch_data,
        batch_sampler=UniformWithReplacementSampler(num_samples=nrows, sample_rate=sample_rate),
        pin_memory=True,
        # batch_size=args.batch_size,
    )
    if not args.non_private_training:
        privacy_engine = PrivacyEngine(
            # secure_rng=args.secure_rng,
            module=model,
            sample_rate=sample_rate,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=args.num_epochs,
            max_grad_norm=args.max_grad_norm,
        )
        # model.privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        # model, optimizer, data_loader = model.privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=data_loader,
        #     epochs=args.num_epochs,
        #     target_epsilon=args.target_epsilon,
        #     target_delta=args.target_delta,
        #     max_grad_norm=args.max_grad_norm,
        # )
        # print(model)
        # print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")
        num_epochs, results = model.train(
            data_loader, args.num_epochs, args.tracked_metrics, privacy_engine=privacy_engine
        )
    else:
        num_epochs, results = model.train(data_loader, args.num_epochs, args.tracked_metrics)

    synthetic = pd.DataFrame(model.generate(nrows), columns=prepared_dataset.columns)
    synthetic = mt.inverse_apply(synthetic)

    fn_output, fn_model = check_output_paths(fn_dataset, args.synthetic, args.model_file, dir_experiment)
    synthetic.to_pickle(dir_experiment / fn_output)
    synthetic.to_csv(dir_experiment / (fn_output[:-3] + "csv"), index=False)
    model.save(dir_experiment / fn_model)

    if "evaluation" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset, "synthetic": synthetic})
    if "plotting" in args.modules_to_run:
        args.module_handover.update({"results": results, "num_epochs": num_epochs})

    print("")

    return args
