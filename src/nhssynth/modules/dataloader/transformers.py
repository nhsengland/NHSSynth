import warnings

import pandas as pd
from nhssynth.modules.dataloader.utils import *
from rdt import HyperTransformer
from rdt.transformers import *
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer


def get_transformer(d: dict):
    # Need to copy in case dicts are shared across columns, this can happen when reading a yaml with anchors
    transformer_data = d.get("transformer").copy()
    if isinstance(transformer_data, dict):
        transformer_name = transformer_data.pop("name", None)
        return eval(transformer_name)(**transformer_data) if transformer_name else None
    elif transformer_data:
        return eval(transformer_data)()
    else:
        return None


def instantiate_synthesizer(sdtypes, transformers, data, allow_null_transformers: bool):
    if all(sdtypes.values()):
        metadata = SingleTableMetadata.load_from_dict({"columns": sdtypes})
    else:
        warnings.warn(
            f"Incomplete metadata, detecting missing `sdtype`s for column(s): {[k for k, v in sdtypes.items() if not v]} automatically...",
            UserWarning,
        )
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        for column_name, values in sdtypes.items():
            if values:
                metadata.update_column(column_name=column_name, **values)
    if not all(transformers.values()) and not allow_null_transformers:
        warnings.warn(
            f"Incomplete metadata, detecting missing `transformers`s for column(s): {[k for k, v in transformers.items() if not v]} automatically...",
            UserWarning,
        )
    # TODO support choice of synthesizer to base this on (not sure how much it matters),
    # synthesizer on some level must form the basis upon which the transformers are inferred from the metadata
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.auto_assign_transformers(data)
    synthesizer.update_transformers(
        transformers if allow_null_transformers else {k: v for k, v in transformers.items() if v}
    )
    return metadata, synthesizer


def instantiate_hypertransformer(sdtypes, transformers, data, allow_null_transformers: bool) -> HyperTransformer:
    ht = HyperTransformer()
    if all(sdtypes.values()) and (all(transformers.values()) or allow_null_transformers):
        ht.set_config(
            config={
                "sdtypes": {k: v["sdtype"] for k, v in sdtypes.items()},
                "transformers": transformers,
            }
        )
    else:
        warnings.warn(
            f"Incomplete metadata, detecting missing{(' `sdtype`s for column(s): ' + str([k for k, v in sdtypes.items() if not v])) if not all(sdtypes.values()) else ''}{(' `transformer`s for column(s): ' + str([k for k, v in transformers.items() if not v])) if not all(transformers.values()) and not allow_null_transformers else ''} automatically...",
            UserWarning,
        )
        ht.detect_initial_config(data)
        ht.update_sdtypes({k: v["sdtype"] for k, v in sdtypes.items() if v})
        ht.update_transformers(
            transformers if allow_null_transformers else {k: v for k, v in transformers.items() if v}
        )
    return ht


# TODO maybe there is a cleaner way of doing this functionality without having the empty dict space for unspecified cols in the data?
def instantiate_metatransformer(
    metadata: dict, data: pd.DataFrame, sdv_workflow: bool, allow_null_transformers: bool
) -> tuple[dict[str, dict], SingleTableMetadata]:

    sdtypes = {cn: filter_inner_dict(cd, {"dtype", "transformer"}) for cn, cd in metadata.items()}
    transformers = {cn: get_transformer(cd) for cn, cd in metadata.items()}
    if sdv_workflow:
        metatransformer = instantiate_synthesizer(sdtypes, transformers, data, allow_null_transformers)
    else:
        metatransformer = instantiate_hypertransformer(sdtypes, transformers, data, allow_null_transformers)
    return metatransformer


def apply_transformer(metatransformer, typed_input, sdv_workflow: bool):
    if sdv_workflow:
        transformed_input = metatransformer.preprocess(typed_input)
    else:
        transformed_input = metatransformer.fit_transform(typed_input)
    return transformed_input
