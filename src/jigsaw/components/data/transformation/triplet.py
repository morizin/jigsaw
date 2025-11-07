from ....core.config_entity import DataTransformationConfig
from ....errors import TransformationError
from ....core import Directory
from ....utils.common import save_csv
from pandas.core.frame import DataFrame
from tqdm.auto import tqdm
from typeguard import typechecked
from .... import logger
import pandas as pd


@typechecked
def triplet_dataset(
    config: DataTransformationConfig,
    data: pd.DataFrame,
    dataname: str,
    filename: str,
    outdir: Directory | None = None,
) -> DataFrame:
    triplet_config = config.triplet
    schema = config.schemas[dataname]

    if isinstance(schema.target, list) and len(schema.target) == 1:
        target = schema.target[0]
    elif isinstance(schema.target, str):
        target = schema.target
    else:
        e = TransformationError(
            "Triplet Conversion incompatible : Invalid target",
            dataname=dataname,
            file_name=filename,
        )
        logger.error(e)
        raise e

    if filename == "sample_submission.csv":
        return data

    if any([col not in data.columns for col in schema.features + [target]]):
        return data
    else:
        data = data[schema.features + [target]]
    if data[target].values[0] == 0:
        ((_, neg), (_, pos)) = data.groupby(target)
    else:
        ((_, pos), (_, neg)) = data.groupby(target)

    if triplet_config.reversed:
        pos, neg = neg, pos

    pos = pos.reset_index(drop=True)
    neg = neg.reset_index(drop=True)

    anchors = (
        pos.groupby(triplet_config.anchor_col)
        .apply(lambda x: list(x.sample(len(x)).index))
        .to_dict()
    )

    neg_repeat = pd.concat([neg] * triplet_config.n_samples, axis=0).reset_index(
        drop=True
    )
    negatives = []

    logger.info(
        f"Generating {triplet_config.n_samples}x samples with {triplet_config.n_negatives} {'positives' if triplet_config.reversed else 'negatives'} each in the file {dataname}.{filename}"
    )
    for idx, negative in tqdm(neg_repeat.iterrows(), total=len(neg_repeat)):
        chosen_idx = []
        remaining = triplet_config.n_negatives

        if remaining:
            anchor = anchors[negative.rule]
            idx = min(remaining, len(anchor))
            chosen_idx.extend(anchor[:idx])
            anchors[negative[triplet_config.anchor_col]] = anchor[idx:]
            remaining -= idx

        count = 0
        while remaining > 0:
            # if remaining:
            anchors = (
                pos.groupby(triplet_config.anchor_col)
                .apply(lambda x: list(x.sample(len(x)).index))
                .to_dict()
            )
            anchor = anchors[negative.rule]
            idx = min(remaining, len(anchor))
            chosen_idx.extend(anchor[:idx])
            anchors[negative.rule] = anchor[idx:]
            remaining -= idx
            count += 1

        negatives.append(chosen_idx)

    negatives = pd.DataFrame(
        [pos.loc[idx, triplet_config.sample_col].values for idx in negatives],
        columns=[f"negative_{i}" for i in range(len(negatives[0]))],
        index=range(len(negatives)),
    )
    assert negatives.shape == (
        neg.shape[0] * triplet_config.n_samples,
        triplet_config.n_negatives,
    ), logger.error(
        f"Error when generating triplets '{dataname}.{filename}', shape doesn't match {negatives.shape} != ({neg.shape[0] * triplet_config.n_samples}, {triplet_config.n_negatives}"
    )
    neg_repeat = pd.concat([neg_repeat, negatives], axis=1)
    neg_repeat = neg_repeat.drop_duplicates(ignore_index=True)
    neg_repeat = neg_repeat.drop(schema.target, axis=1)
    neg_repeat = neg_repeat.rename(
        columns={
            triplet_config.anchor_col: "anchor",
            triplet_config.sample_col: "positive",
        }
    )
    if triplet_config.n_negatives == 1:
        neg_repeat = neg_repeat.rename(columns={"negative_0": "negative"})

    if outdir:
        save_csv(data, outdir // f"triplet_{dataname}" / filename)
    return neg_repeat
