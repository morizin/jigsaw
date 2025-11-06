import pandas as pd
from ..data.transformation.triplet import triplet_dataset
from ..data import get_data
from ...utils.common import load_csv
from .prompts import zero_shot_chat_prompt
from datasets import Dataset
from ..data.augmentation import do_aug
import os


def create_pseudo_labels(row, margin=0.2):
    if row > 1 - margin:
        return 1
    if row < margin:
        return 0
    return -1


def get_train_test_split(
    training_config, tokenizer=None, config=None, transform_config=None, preprocess=None
):
    if config is None:
        raise ValueError("Config Not Given")
    train_data, valid_data, test_data = get_data(training_config, config, preprocess)
    print("before train samples", train_data.shape)

    if "train_frac" in config:
        # train_data = [train_data]
        # mlskf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1234 + random.randint(12, 1200)).split(test_data, test_data['rule_violation'])
        # for _ in range(int(config['train_frac']*100)):
        # train_data.append(test_data.iloc[next(mlskf)[1], :])
        train_data = pd.concat(
            [train_data, test_data.sample(frac=config["train_frac"])], axis=0
        ).reset_index(drop=True)
    # train_data = train_data.sample(frac = 0.8)

    if "triplet" in config["model_type"].lower().strip():
        if transform_config is None:
            raise ValueError("DataTransformConfig is not provided")

        train_data = triplet_dataset(
            transform_config, train_data, ["raw", "test.csv"], ""
        )
        valid_data = None
        return Dataset.from_pandas(train_data), valid_data

    if "psuedo_file" in config and config["psuedo_file"]:
        psuedo_file = os.path.join(
            "/kaggle/working/", config["psuedo_file"], "submission.csv"
        )
        if os.path.exists(psuedo_file):
            print("adding Psuedo labels ...")
            test = load_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
            test = pd.merge(test, load_csv(psuedo_file), on="row_id", how="right")
            test["rule_violation"] = test["rule_violation"].apply(
                create_pseudo_labels,
                args=(config["psuedo_margin"] if "psuedo_margin" in config else 0.2,),
            )
            test = test.query("rule_violation != -1")
            train_data = pd.concat(
                [train_data, test[train_data.columns]], axis=0
            ).reset_index(drop=True)

    if config.get("do_aug", None):
        train_data = do_aug(train_data, config)

    train_data["prompt"] = train_data.apply(
        zero_shot_chat_prompt, args=(tokenizer,), axis=1
    )
    train_data["rule_violation"] = train_data["rule_violation"].map(
        {
            1: " Yes",
            0: " No",
        }
    )
    train_data["prompt"] += train_data["rule_violation"]

    if valid_data is not None:
        valid_data["prompt"] = valid_data.apply(
            zero_shot_chat_prompt, args=(tokenizer,), axis=1
        )
        valid_data["rule_violation"] = valid_data["rule_violation"].map(
            {
                1: " Yes",
                0: " No",
            }
        )
        valid_data["prompt"] += valid_data["rule_violation"]

    train_data["input_ids"] = train_data["prompt"].apply(
        lambda x: tokenizer(x, add_special_tokens=False, truncation=False)["input_ids"]
    )

    return train_data, valid_data
