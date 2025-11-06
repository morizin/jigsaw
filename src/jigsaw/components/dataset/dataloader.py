from typeguard import typechecked
import torch


def collate_fn(batch_input: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    inputs = {
        "input_ids": torch.stack([inp["input_ids"] for inp in batch_input], dim=0),
        "attention_mask": torch.stack(
            [inp["attention_mask"] for inp in batch_input], dim=0
        ),
        "token_type_ids": torch.stack(
            [inp["token_type_ids"] for inp in batch_input], dim=0
        ),
        "labels": torch.cat([inp["labels"] for inp in batch_input]),
    }

    return inputs
