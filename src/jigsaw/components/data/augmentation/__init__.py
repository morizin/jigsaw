from typing import Callable, Collection
from pydantic import BaseModel, field_validator, validate_call, model_validator
import random
from .augment_utils import (
    url_cleaner,
    url_to_semantics,
    RandomURL,
    transileration,
    sentence_jumbling,
    random_sentence,
    rule_flip,
    TfidfAug,
)
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import pandas as pd
import inspect

AUGMENT_DICT = {
    "ocr": nac.OcrAug().augment,
    "keystroke": nac.KeyboardAug().augment,
    "random_char_substitute": nac.RandomCharAug(action="substitute").augment,
    "random_char_insert": nac.RandomCharAug(action="insert").augment,
    "random_char_swap": nac.RandomCharAug(action="swap").augment,
    "random_char_delete": nac.RandomCharAug(action="delete").augment,
    "spelling": naw.SpellingAug().augment,
    "random_word_swap": naw.RandomWordAug(action="swap").augment,
    "url_cleaner": url_cleaner,
    "url_to_semantics": url_to_semantics,
    "random_url": RandomURL,
    "transileration": transileration,
    "sentence_shuffle": sentence_jumbling,
    "random_gen_sentence": random_sentence,
    "rule_flip": rule_flip,
    "tfidf": TfidfAug,
}


class Augment(BaseModel):
    augment: Callable
    p: float = 1.0

    @field_validator("p", mode="before")
    @classmethod
    def probability(cls, v: float | int) -> float:
        if isinstance(v, int):
            v /= 100
        return v

    @field_validator("augment", mode="before")
    @classmethod
    def get_augment(cls, augment_name: str) -> Callable:
        if AUGMENT_DICT.get(augment_name, False):
            return AUGMENT_DICT[augment_name]
        else:
            raise

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


class Augmentor(BaseModel):
    augments: list[Augment]
    is_tta: bool = False
    frac: float = 0.1
    resample: int = 2
    include_original: bool = True
    weight: float = 0.25

    @field_validator("augments", mode="before")
    @classmethod
    def get_augment(cls, augment_list: Collection[Collection]):
        return [Augment(augment=aug_name, prob=prob) for aug_name, prob in augment_list]

    @field_validator("frac", mode="before")
    @classmethod
    def set_frac(cls, frac: float | int) -> float:
        if isinstance(frac, int):
            frac /= 100
        return frac

    @model_validator(mode="after")
    def set_weight(self) -> "Augmentor":
        self.weight /= self.resample
        return self

    def augment(self, data):
        augs = []
        if self.include_original:
            augs.append(data)

        for _ in range(self.resample):
            temp = data.copy()
            for aug in self.augments:
                temp = temp.apply(aug.apply, axis=1)

            if self.is_tta and "weight" in data.columns:
                temp["weight"] = self.weight

            augs.append(
                (temp.sample(frac=self.frac) if self.include_original else temp)[
                    data.columns
                ]
            )
        return (
            pd.concat(augs, axis=0)
            .reset_index(drop=True)
            .drop_duplicates(subset=["rule", "body"], ignore_index=True)
        )
