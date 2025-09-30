from ..constants.prompt import *
from typeguard import typechecked

@typechecked
def build_prompt(row):
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
