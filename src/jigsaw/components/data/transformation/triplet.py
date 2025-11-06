from ...schema.config_entity import DataTransformationConfig
from ....core import FilePath, Directory
from ....utils.common import save_csv
from jigsaw.components.data.transformation.cleaning import remove_duplicates
from pandas.core.frame import DataFrame
from tqdm.auto import tqdm
from typeguard import typechecked
from jigsaw import logger
import pandas as pd
from pathlib import Path


@typechecked
def triplet_dataset(
    config: DataTransformationConfig,
    data: DataFrame,
    path: list[str],
    name: str,
    outdir: FilePath | None = None,
) -> DataFrame:
    dataname, filename = path
    triplet_config = config.triplet

    if filename == "sample_submission.csv":
        return data

    if set(data.columns) != set(config.features[dataname] + [config.targets[dataname]]):
        return data

    if data.rule_violation.values[0] == 0:
        ((_, neg), (_, pos)) = data.groupby("rule_violation")
    else:
        ((_, pos), (_, neg)) = data.groupby("rule_violation")

    pos = pos.reset_index(drop=True)
    neg = neg.reset_index(drop=True)

    rules = pos.groupby("rule").apply(lambda x: list(x.sample(len(x)).index)).to_dict()

    neg_repeat = pd.concat([neg] * triplet_config.ntriplets, axis=0).reset_index(
        drop=True
    )
    negatives = []

    logger.info(
        f"Generating {triplet_config.nsamples} samples of {triplet_config.ntriplets} triplets each in the file {dataname}.{filename}"
    )
    for idx, negative in tqdm(neg_repeat.iterrows(), total=len(neg_repeat)):
        # subred = sr.get((positive.rule, positive.subreddit), None)
        chosen_idx = []
        remaining = triplet_config.nsamples
        # if subred:
        #     idx = min(len(subred), remaining)
        #     chosen_idx.extend(subred[:idx])
        #     sr[(positive.rule, positive.subreddit)] = subred[idx:]
        #     remaining -= idx

        if remaining:
            rule = rules[negative.rule]
            idx = min(remaining, len(rule))
            chosen_idx.extend(rule[:idx])
            rules[negative.rule] = rule[idx:]
            remaining -= idx

        while remaining > 0:
            # if remaining:
            rules = (
                pos.groupby("rule")
                .apply(lambda x: list(x.sample(len(x)).index))
                .to_dict()
            )
            rule = rules[negative.rule]
            idx = min(remaining, len(rule))
            chosen_idx.extend(rule[:idx])
            rules[negative.rule] = rule[idx:]
            remaining -= idx

        # if remaining:
        #     raise

        negatives.append(chosen_idx)

    negatives = pd.DataFrame(
        [pos.loc[idx, "body"].values for idx in negatives],
        columns=[f"negative_{i}" for i in range(len(negatives[0]))],
        index=range(len(negatives)),
    )
    # print(negatives, neg_repeat)
    assert negatives.shape == (
        neg.shape[0] * triplet_config.ntriplets,
        triplet_config.nsamples,
    ), logger.error(
        f"Error when generating triplets '{dataname}.{filename}', shape doesn't match {negatives.shape} != ({pos.shape[0] * triplet_config.ntriplets}, {triplet_config.nsamples}"
    )

    logger.info(
        f"Generating {triplet_config.nsamples} samples of {triplet_config.ntriplets} triplets each in the file {dataname}.{filename}"
    )
    neg_repeat = pd.concat([neg_repeat, negatives], axis=1)
    # neg_repeat = remove_duplicates(config, neg_repeat, path, name)
    neg_repeat = neg_repeat.drop_duplicates(ignore_index=True)
    neg_repeat = neg_repeat.drop("rule_violation", axis=1)
    neg_repeat = neg_repeat.rename(columns={"rule": "anchor", "body": "positive"})
    if triplet_config.nsamples == 1:
        neg_repeat = neg_repeat.rename(columns={"negative_0": "negative"})

    if outdir:
        target_dir = Directory(
            path=(Path(outdir) if isinstance(outdir, str) else outdir) / name
        )
        save_csv(neg_repeat, target_dir.path / filename)
    return neg_repeat
