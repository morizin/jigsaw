from ...entity.config_entity import DataTransformationConfig
from ...entity.common import Directory
from ...utils.common import save_csv
from .cleaning import remove_duplicates
from pandas.core.frame import DataFrame
from ensure import ensure_annotations
from tqdm.autonotebook import tqdm
from ... import logger
import pandas as pd
import numpy as np
from pathlib import Path
import random

@ensure_annotations
def triplet_dataset(config: DataTransformationConfig,
                    data: DataFrame, 
                    path : list,
                    name : str,
                    outdir: str | Path | None = None) -> DataFrame:

    dataname, filename = path
    triplet_config = config.triplet

    if filename == 'sample_submission.csv':
        return data
    
    if set(data.columns) != set(config.features[dataname] + [config.targets[dataname]]):
        return data
    
    if data.loc[0, 'rule_violation'] == 0:
        ((_, neg), (_, pos)) = data.groupby('rule_violation')
    else:
        ((_, pos), (_, neg)) = data.groupby("rule_violation")

    pos = pos.reset_index(drop = True)
    neg = neg.reset_index(drop = True)

    rules = neg.groupby('rule').apply(
        lambda x: list(
            x.sample(
                len(x)
            ).index
        )
    ).to_dict()

    sr = neg.groupby(['rule', 'subreddit']).apply(
        lambda x: list(
            x.sample(
                len(x)
            ).index
        )
    ).to_dict()

    pos_repeat = pd.concat([pos] * triplet_config.ntriplets, axis = 0)
    negatives = []

    logger.info(f"Generating {triplet_config.nsamples} samples of {triplet_config.ntriplets} triplets each in the file {dataname}.{filename}")
    for idx, positive in tqdm(pos_repeat.iterrows(), total = len(pos_repeat)):
        subred = sr.get((positive.rule, positive.subreddit), None)
        chosen_idx = []
        remaining = triplet_config.nsamples
        if subred:
            idx = min(len(subred), remaining)
            chosen_idx.extend(subred[:idx])
            sr[(positive.rule, positive.subreddit)] = subred[idx:]
            remaining -= idx

        if remaining:
            rule = rules[positive.rule]
            idx = min(remaining, len(rule))
            chosen_idx.extend(rule[:idx]) 
            rules[positive.rule] = rule[idx:]
            remaining -= idx
            
        while remaining > 0:
            rules = neg.groupby('rule').apply(
                lambda x: list(
                    x.sample(
                        len(x)
                    ).index
                )
            ).to_dict()
            rule = rules[positive.rule]
            idx = min(remaining, len(rule))
            chosen_idx.extend(rule[:idx]) 
            rules[positive.rule] = rule[idx:]
            remaining -= idx

        negatives.append(chosen_idx)

    negatives = pd.DataFrame([neg.loc[idx, 'body'].values for idx in negatives], columns = [f"negative_{i}" for i in range(len(negatives[0]))], index = range(len(negatives))) 

    assert negatives.shape == (pos.shape[0] * triplet_config.ntriplets, triplet_config.nsamples), logger.error(f"Error when generating triplets '{dataname}.{filename}', shape doesn't match {negatives.shape} != ({pos.shape[0] * triplet_config.ntriplets}, {triplet_config.nsamples}")

    logger.info(f"Generating {triplet_config.nsamples} samples of {triplet_config.ntriplets} triplets each in the file {dataname}.{filename}")

    pos_repeat = pd.concat([pos_repeat, negatives], axis = 1)
    pos_repeat = remove_duplicates(config, pos_repeat, path, name)
    pos_repeat = pos_repeat.drop('rule_violation', axis = 1)
    
    if outdir:
        target_dir = Directory(path = (Path(outdir) if isinstance(outdir, str) else outdir) / name)
        save_csv(pos_repeat, target_dir.path / filename)
    return pos_repeat 
