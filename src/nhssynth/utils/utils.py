from typing import Any

import numpy as np
import pandas as pd
import torch


def filter_dict(d: dict, excludes: set) -> dict:
    """
    Returns a new dictionary containing all key-value pairs from the input dictionary `d`, except those whose key is
    included in the set `excludes`.

    Args:
        d (dict): The input dictionary to filter.
        excludes (set): A set containing the keys to exclude from the filtered dictionary.

    Returns:
        dict: A new dictionary containing all key-value pairs from `d` except those whose key is included in `excludes`.
    """
    return {k: v for k, v in d.items() if k not in excludes}


def get_key_by_value(d: dict[Any, Any], value: Any) -> Any | None:
    """
    Find the first key in a dictionary with a given value.

    Args:
        d: A dictionary to search through.
        value: The value to search for.

    Returns:
        The first key in `d` with the value `value`, or `None` if no such key exists.

    Examples:
        >>> d = {'a': 1, 'b': 2, 'c': 1}
        >>> get_key_by_value(d, 2)
        'b'
        >>> get_key_by_value(d, 3)
        None

    """
    for key, val in d.items():
        if val == value:
            return key
    return None


def flatten_dict(d: dict[str, Any | dict]) -> dict[str, Any]:
    """
    Flatten a dictionary by recursively combining nested keys into a single dictionary until no nested keys remain.

    Args:
        d: A dictionary with possibly nested keys.

    Returns:
        A flattened dictionary.

    Examples:
        >>> d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> flatten_dict(d)
        {'a': 1, 'c': 2, 'e': 3}
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# # -------- Reverse Transformations -------- #


# def reverse_transformers(
#     synthetic_set,
#     data_supp_columns,
#     cont_transformers=None,
#     cat_transformers=None,
#     date_transformers=None,
#     pre_proc_method="GMM",
# ):

#     # Now all of the transformations from the dictionary - first loop over the categorical columns

#     synthetic_transformed_set = synthetic_set.copy(deep=True)

#     if cat_transformers != None:
#         for transformer_name in cat_transformers:

#             transformer = cat_transformers[transformer_name]
#             column_name = transformer_name[12:]

#             synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

#     if cont_transformers != None:

#         if pre_proc_method == "GMM":

#             for transformer_name in cont_transformers:

#                 transformer = cont_transformers[transformer_name]
#                 column_name = transformer_name[11:]

#                 synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

#         elif pre_proc_method == "standard":

#             for transformer_name in cont_transformers:

#                 transformer = cont_transformers[transformer_name]
#                 column_name = transformer_name[11:]

#                 # Reverse the standard scaling
#                 synthetic_transformed_set[column_name] = transformer.inverse_transform(
#                     synthetic_transformed_set[column_name].values.reshape(-1, 1)
#                 ).flatten()

#     if date_transformers != None:
#         for transformer_name in date_transformers:

#             transformer = date_transformers[transformer_name]
#             column_name = transformer_name[9:]

#             synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

#     synthetic_transformed_set = pd.DataFrame(synthetic_transformed_set, columns=data_supp_columns)

#     return synthetic_transformed_set


# def plot_elbo(
#     n_epochs,
#     log_elbo,
#     log_reconstruction,
#     log_divergence,
#     saving_filepath=None,
#     pre_proc_method="GMM",
# ):

#     x = np.arange(n_epochs)

#     y1 = log_elbo
#     y2 = log_reconstruction
#     y3 = log_divergence

#     plt.plot(x, y1, label="ELBO")
#     plt.plot(x, y2, label="RECONSTRUCTION")
#     plt.plot(x, y3, label="DIVERGENCE")
#     plt.xlabel("Number of Epochs")
#     # Set the y axis label of the current axis.
#     plt.ylabel("Loss Value")
#     # Set a title of the current axes.
#     plt.title("ELBO Breakdown")
#     # show a legend on the plot
#     plt.legend()

#     if saving_filepath != None:
#         # Save static image
#         plt.savefig("{}ELBO_Breakdown_SynthVAE_{}.png".format(saving_filepath, pre_proc_method))

#     plt.show()

#     return None


# def plot_likelihood_breakdown(
#     n_epochs,
#     log_categorical,
#     log_numerical,
#     saving_filepath=None,
#     pre_proc_method="GMM",
# ):

#     x = np.arange(n_epochs)

#     y1 = log_categorical
#     y2 = log_numerical

#     plt.subplot(1, 2, 1)
#     plt.plot(x, y1, label="CATEGORICAL")
#     plt.xlabel("Number of Epochs")
#     # Set the y axis label of the current axis.
#     plt.ylabel("Loss Value")
#     # Set a title of the current axes.
#     plt.title("Categorical Breakdown")
#     # show a legend on the plot
#     plt.subplot(1, 2, 2)
#     plt.plot(x, y2, label="NUMERICAL")
#     plt.xlabel("Number of Epochs")
#     # Set the y axis label of the current axis.
#     plt.ylabel("Loss Value")
#     # Set a title of the current axes.
#     plt.title("Numerical Breakdown")
#     # show a legend on the plot
#     plt.tight_layout()

#     if saving_filepath != None:
#         # Save static image
#         plt.savefig("{}Reconstruction_Breakdown_SynthVAE_{}.png".format(saving_filepath, pre_proc_method))

#     return None


# def plot_variable_distributions(
#     categorical_columns,
#     continuous_columns,
#     data_supp,
#     synthetic_supp,
#     saving_filepath=None,
#     pre_proc_method="GMM",
# ):

#     # Plot some examples using plotly

#     for column in categorical_columns:

#         plt.subplot(1, 2, 1)
#         plt.hist(x=synthetic_supp[column])
#         plt.title("Synthetic")
#         # Set the x axis label of the current axis
#         plt.xlabel("Data Value")
#         # Set the y axis label of the current axis.
#         plt.ylabel("Distribution")
#         # Set a title of the current axes.
#         plt.title("Synthetic".format(column))
#         # show a legend on the plot
#         plt.subplot(1, 2, 2)
#         plt.hist(x=data_supp[column])
#         plt.title("Original")
#         # Set the x axis label of the current axis
#         plt.xlabel("Data Value")
#         # Set the y axis label of the current axis.
#         plt.ylabel("Distribution")
#         # Set a title of the current axes.
#         plt.title("Original".format(column))
#         # show a legend on the plot
#         plt.suptitle("Variable {}".format(column))

#         plt.tight_layout()

#         if saving_filepath != None:
#             # Save static image
#             plt.savefig("{}Variable_{}_SynthVAE_{}.png".format(saving_filepath, column, pre_proc_method))

#         plt.show()

#     for column in continuous_columns:

#         plt.subplot(1, 2, 1)
#         plt.hist(x=synthetic_supp[column])
#         plt.title("Synthetic")
#         # Set the x axis label of the current axis
#         plt.xlabel("Data Value")
#         # Set the y axis label of the current axis.
#         plt.ylabel("Distribution")
#         # Set a title of the current axes.
#         plt.title("Synthetic".format(column))
#         # show a legend on the plot
#         plt.subplot(1, 2, 2)
#         plt.hist(x=data_supp[column])
#         plt.title("Original")
#         # Set the x axis label of the current axis
#         plt.xlabel("Data Value")
#         # Set the y axis label of the current axis.
#         plt.ylabel("Distribution")
#         # Set a title of the current axes.
#         plt.title("Original".format(column))
#         # show a legend on the plot
#         plt.suptitle("Variable {}".format(column))

#         plt.tight_layout()

#         if saving_filepath != None:
#             # Save static image
#             plt.savefig("{}Variable_{}_SynthVAE_{}.png".format(saving_filepath, column, pre_proc_method))

#         plt.show()

#         return None
