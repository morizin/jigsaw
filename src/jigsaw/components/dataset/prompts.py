from ...constants.prompt import (
    COMPLETION_PHRASE,
    SYSTEM_PROMPT,
    POSITIVE_ANSWER,
    NEGATIVE_ANSWER,
)
from typeguard import typechecked


@typechecked
def few_shot_completion_prompt(row, *args):
    return f"""{SYSTEM_PROMPT}
r/{row.subreddit}
Rule: {row.rule}

1) {row.positive_example_1}
{COMPLETION_PHRASE}: {POSITIVE_ANSWER}

2) {row.negative_example_1}
{COMPLETION_PHRASE}: {NEGATIVE_ANSWER}

3) {row.positive_example_2}
{COMPLETION_PHRASE}: {POSITIVE_ANSWER}

4) {row.negative_example_2}
{COMPLETION_PHRASE}: {NEGATIVE_ANSWER}

5) {row.body}
{COMPLETION_PHRASE}: """


@typechecked
def few_shot_chat_prompt(row, tokenizer):
    SYS_PROMPT = f"""
You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond {POSITIVE_ANSWER} or {NEGATIVE_ANSWER}.
Rule : {row.rule.strip()}
"""
    messages = [
        {"role": "system", "content": SYS_PROMPT.strip()},
        {"role": "user", "content": row.positive_example_1.strip()},
        {"role": "assistant", "content": POSITIVE_ANSWER.strip()},
        {"role": "user", "content": row.negative_example_1.strip()},
        {"role": "assistant", "content": NEGATIVE_ANSWER.strip()},
        {"role": "user", "content": row.positive_example_2.strip()},
        {"role": "assistant", "content": POSITIVE_ANSWER.strip()},
        {"role": "user", "content": row.negative_example_2.strip()},
        {"role": "assistant", "content": NEGATIVE_ANSWER.strip()},
        {"role": "user", "content": row.body.strip()},
    ]

    messages = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return messages


def zero_shot_chat_prompt(row, tokenizer):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip() + f"Rule : {row.rule.strip()}",
        },
        {"role": "user", "content": row.body.strip()},
    ]

    messages = (
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        + f"{COMPLETION_PHRASE}:"
    )

    return messages
