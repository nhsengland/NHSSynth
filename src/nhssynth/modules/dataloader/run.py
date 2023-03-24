import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.io import *
from nhssynth.modules.dataloader.metadata import *
from nhssynth.modules.dataloader.transformers import *


def run(args) -> None:

    if args.seed:
        np.random.seed(args.seed)

    input_path, output_path, metadata_input_path, metadata_output_path, experiment_path = format_io(
        args.input_file, args.output_file, args.metadata_file, args.dir, run_name=args.run_name
    )
    experiment_path.mkdir(parents=True, exist_ok=True)

    input = pd.read_csv(input_path, index_col=args.index_col)
    metadata = load_metadata(metadata_input_path, input)

    dtypes = instantiate_dtypes(metadata, input)
    # TODO point out when this fails that it must be due to an invalid / unsupported dtype in the metadata
    typed_input = input.astype(dtypes)

    metatransformer = instantiate_metatransformer(
        metadata, typed_input, args.sdv_workflow, args.allow_null_transformers
    )

    output_metadata(metadata_output_path, dtypes, metatransformer, args.sdv_workflow, args.collapse_yaml)

    transformed_input = apply_transformer(metatransformer, typed_input, args.sdv_workflow)
    transformed_input.to_csv(output_path, index=False)
