from typing import Callable
from pydantic import BaseModel, field_validator, validate_call
import random
from .augment_utils import *
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import pandas as pd
import inspect


class Augment(BaseModel):
    augment: Callable
    p: float

    @field_validator("p", mode="before")
    @classmethod
    @validate_call
    def probability(cls, v: float | int) -> float:
        if isinstance(v, int):
            v /= 100
        return v

    def apply(self, row):
        if random.random() < self.p:
            if "row" in inspect.signature(self.augment).parameters:
                return self.augment(row)
            text = self.augment(row["body"])
            if isinstance(text, list):
                if len(text) > 0:
                    text = text[0]
                else:
                    text = row.body
            row.body = text
        return row


# Now we need to use
def do_aug(data, config, tta=False):
    print("Augmenting data...")
    if tta:
        aug_args = config["do_tta"]
    else:
        aug_args = config["do_aug"]
    print(f"{aug_args = }")
    p = aug_args[0] or 0.1
    n = aug_args[1] or 2
    append = aug_args[2] if (len(aug_args) > 2 and aug_args[2] is not None) else True

    AUGMENTS = [
        Augment(augment=nac.OcrAug().augment, p=p),
        Augment(augment=nac.KeyboardAug().augment, p=p),
        # Augment(augment = nac.RandomCharAug(action="substitute").augment, p = p),
        Augment(augment=nac.RandomCharAug(action="insert").augment, p=p),
        # Augment(augment = nac.RandomCharAug(action="swap").augment, p = p),
        # Augment(augment = nac.RandomCharAug(action="delete").augment, p = p),
        Augment(augment=naw.SpellingAug().augment, p=p),
        Augment(augment=naw.RandomWordAug(action="swap").augment, p=p),
        Augment(augment=naw.RandomWordAug(action="crop").augment, p=p),
        # Augment(augment = naw.RandomWordAug(action="substitute").augment, p = p),
        # Augment(augment = naw.RandomWordAug(action="delete").augment, p = p),
        Augment(augment=url_cleaner, p=p),
        Augment(augment=url_to_semantics, p=p),
        Augment(augment=RandomURL(data), p=p),
        Augment(augment=transileration, p=p),
        Augment(augment=sentence_jumbling, p=p),
        Augment(augment=random_sentence, p=p),
        Augment(augment=rule_flip, p=p + 0.25),
        Augment(augment=TfidfAug(data), p=p),
    ]

    augs = []
    if append:
        augs.append(data)

    aug_weight = (aug_args[3] if len(aug_args) > 3 else 0.25) / n
    for _ in range(n):
        temp = data.copy()
        for aug in AUGMENTS:
            temp = temp.apply(aug.apply, axis=1)

        if tta and "weight" in data.columns:
            temp["weight"] = aug_weight
        augs.append((temp.sample(frac=p) if append else temp)[data.columns])
    del AUGMENTS
    return (
        pd.concat(augs, axis=0)
        .reset_index(drop=True)
        .drop_duplicates(subset=["rule", "body"], ignore_index=True)
    )
