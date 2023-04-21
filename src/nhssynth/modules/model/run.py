import argparse
import warnings

import pandas as pd
import torch
from nhssynth.common import *
from nhssynth.modules.model.DPVAE import VAE, Decoder, Encoder
from nhssynth.modules.model.io import check_output_paths, load_required_data
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset


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
    encoder = Encoder(input_dim=ncols, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    decoder = Decoder(args.latent_dim, onehots, singles)
    e_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    d_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    data_loader = DataLoader(
        torch_data,
        pin_memory=True,
        batch_size=args.batch_size,
    )
    if not args.non_private_training:
        if not args.target_delta:
            args.target_delta = 1 / nrows
        privacy_engine = PrivacyEngine(secure_mode=args.secure_mode)
        # The below raises two warnings, one about log calculation which is resolved by a PR, and the other
        # is due to default alphas being used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
            decoder, d_optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=decoder,
                optimizer=d_optimizer,
                data_loader=data_loader,
                epochs=args.num_epochs,
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                max_grad_norm=args.max_grad_norm,
            )
        print(f"Using sigma={d_optimizer.noise_multiplier} and C={args.max_grad_norm}")
        model = VAE(encoder, decoder, e_optimizer, d_optimizer, onehots, singles, args.use_gpu)
        num_epochs, results = model.train(
            data_loader,
            args.num_epochs,
            tracked_metrics=args.tracked_metrics,
            privacy_engine=privacy_engine,
            target_delta=args.target_delta,
            patience=args.patience,
        )
    else:
        model = VAE(encoder, decoder, e_optimizer, d_optimizer, onehots, singles, args.use_gpu)
        num_epochs, results = model.train(
            data_loader,
            args.num_epochs,
            tracked_metrics=args.tracked_metrics,
            patience=args.patience,
        )

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
