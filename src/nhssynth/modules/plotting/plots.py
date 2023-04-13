# import argparse
# import numpy as np
# import pandas as pd
# import torch

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.manifold import TSNE


def factorize_all_categoricals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Factorize all categorical columns in a dataframe."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]
        elif df[col].dtype == "datetime64[ns]":
            df[col] = pd.to_numeric(df[col])
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)

    return df


def tsne(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> None:
    tsne_real = TSNE(n_components=2, init="pca")
    proj_real = pd.DataFrame(tsne_real.fit_transform(factorize_all_categoricals(real)))

    tsne_synth = TSNE(n_components=2, init="pca")
    proj_synth = pd.DataFrame(tsne_synth.fit_transform(factorize_all_categoricals(synth)))

    fig = go.Figure()

    fig.add_scatter(x=proj_real[0], y=proj_real[1], mode="markers", marker=dict(size=5), opacity=0.75, name="Real data")
    fig.add_scatter(
        x=proj_synth[0], y=proj_synth[1], mode="markers", marker=dict(size=5), opacity=0.75, name="Synthetic data"
    )

    # Set axis labels and legend
    fig.update_layout(
        title="t-SNE Plot",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend=dict(x=0, y=1, bgcolor="rgba(255, 255, 255, 0.5)"),
    )

    # Show plot
    fig.show()


# # For Gower distance
# import gower

# # For the SUPPORT dataset
# from pycox.datasets import support

# # VAE functions
# from VAE import Decoder, Encoder, VAE

# from utils import support_pre_proc, reverse_transformers

# # Plotting
# import matplotlib

# font = {"size": 14}
# matplotlib.rc("font", **font)
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # For the SUPPORT dataset
# from pycox.datasets import support

# # VAE functions
# from VAE import Decoder, Encoder, VAE

# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--save_file",
#     required=True,
#     type=str,
#     help="load trained model's state_dict from file",
# )

# parser.add_argument(
#     "--pre_proc_method",
#     default="GMM",
#     type=str,
#     help="Choose the pre-processing method that you will apply to the dataset, either GMM or standard",
# )

# args = parser.parse_args()

# # Import and preprocess the SUPPORT data for ground truth correlations
# data_supp = support.read_df()

# # Save the original columns

# original_continuous_columns = ["duration"] + [f"x{i}" for i in range(7, 15)]
# original_categorical_columns = ["event"] + [f"x{i}" for i in range(1, 7)]

# original_columns = original_categorical_columns + original_continuous_columns
# #%% -------- Data Pre-Processing -------- #
# pre_proc_method = args.pre_proc_method

# (
#     x_train,
#     data_supp,
#     reordered_dataframe_columns,
#     continuous_transformers,
#     categorical_transformers,
#     num_categories,
#     num_continuous,
# ) = support_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)


# ###############################################################################

# # Load saved model - ensure parameters are equivalent to the saved model
# latent_dim = 256
# hidden_dim = 256
# encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
# decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)
# vae = VAE(encoder, decoder)
# vae.load(args.save_file)

# #%% -------- Generate Synthetic Data -------- #

# # Generate a synthetic set using trained vae

# synthetic_trial = vae.generate(data_supp.shape[0])  # 8873 is size of support
# #%% -------- Inverse Transformation On Synthetic Trial -------- #

# synthetic_sample = vae.generate(data_supp.shape[0])

# if torch.cuda.is_available():
#     synthetic_sample = pd.DataFrame(
#         synthetic_sample.cpu().detach(), columns=reordered_dataframe_columns
#     )
# else:
#     synthetic_sample = pd.DataFrame(
#         synthetic_sample.detach(), columns=reordered_dataframe_columns
#     )

# # Reverse the transformations

# synthetic_supp = reverse_transformers(
#     synthetic_set=synthetic_sample,
#     data_supp_columns=data_supp.columns,
#     cont_transformers=continuous_transformers,
#     cat_transformers=categorical_transformers,
#     pre_proc_method=pre_proc_method,
# )


# ### Create plots
# # Plot 1: Correlation matrix of original data
# plt.figure()
# ax = plt.gca()
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# im = ax.matshow(data_supp.corr())
# #####
# # Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# #####
# plt.colorbar(im, cax=cax)
# plt.savefig("actual_corr_{}.png".format(pre_proc_method), bbox_inches="tight")
# # Plot 2: Correlation matrix of synthetic data
# plt.figure()
# ax = plt.gca()
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# im = ax.matshow(synthetic_supp.corr())
# #####
# # Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# #####
# plt.colorbar(im, cax=cax)
# plt.savefig("sample_corr_{}.png".format(pre_proc_method), bbox_inches="tight")
# # Plot 3: Difference between real and synth correlation matrices + Gower and RMSE values
# plt.figure()
# g = np.mean(gower.gower_matrix(data_supp, synthetic_supp))
# p = np.sqrt(
#     np.mean((data_supp.corr().to_numpy() - synthetic_supp.corr().to_numpy()) ** 2)
# )
# plt.title(f"Gower Distance = {g:.4f}\n Correlation RMSE = {p:.4f}")
# ax = plt.gca()
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# im = ax.matshow(synthetic_supp.corr() - data_supp.corr())
# #####
# # Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# #####
# plt.colorbar(im, cax=cax)
# plt.savefig("diff_corr_{}.png".format(pre_proc_method), bbox_inches="tight")
