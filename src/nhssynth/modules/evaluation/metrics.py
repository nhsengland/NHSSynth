# import gower
# import numpy as np
# import pandas as pd


# # Distributional metrics - Check distribution differences between synthetic & original dataset as well as how
# # Easy it is to discriminate them i.e. svc detection
# def distribution_metrics(
#     args,
#     data_supp,
#     synthetic_supp,
#     categorical_columns,
# ):
#     # Define lists to contain the metrics achieved

#     metrics = []

#     # Need these in same column order

#     synthetic_supp = synthetic_supp[data_supp.columns]

#     # Now categorical columns need to be converted to objects as SDV infers data
#     # types from the fields and integers/floats are treated as numerical not categorical

#     synthetic_supp[categorical_columns] = synthetic_supp[categorical_columns].astype(object)
#     data_supp[categorical_columns] = data_supp[categorical_columns].astype(object)

#     evals = evaluate(synthetic_supp, data_supp, metrics=distributional_metrics, aggregate=False)

#     # evals is a pandas dataframe of metrics - if we want to add a gower metric then we can
#     # save this separately

#     metrics = np.array(evals["raw_score"])

#     if gower_bool == True:
#         # Find the gower distance
#         metrics = np.append(metrics, np.mean(gower.gower_matrix(data_supp, synthetic_supp)))

#         metrics = pd.DataFrame(data=[metrics], columns=(distributional_metrics + ["Gower"]))

#     else:
#         metrics = pd.DataFrame(data=[metrics], columns=(distributional_metrics))

#     # Save these metrics into a pandas dataframe - if the user wants to

#     if saving_filepath != None:
#         metrics.to_csv("{}Metrics_SynthVAE_{}.csv".format(saving_filepath, pre_proc_method))

#     return metrics
