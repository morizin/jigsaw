POSITIVE_ANSWER = "YES"
NEGATIVE_ANSWER = "NO"

SYSTEM_PROMPT = f"""
You are given a comment from reddit and a rule. Your task is classify if it violates the given rule. Only respond {POSITIVE_ANSWER} or {NEGATIVE_ANSWER}.
"""

COMPLETION_PHRASE = "Violation"
