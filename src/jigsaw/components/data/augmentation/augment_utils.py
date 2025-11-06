import re
from urllib.parse import urlparse
import random, itertools
import pronouncing
import numpy as np
import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw


class TfidfAug:
    def __init__(self, data, label=0):
        self.label = label
        train_x_tokens = [self._tokenizer(x) for x in data.body]
        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save(".")
        self.aug = naw.TfIdfAug(model_path=".", tokenizer=self._tokenizer)

    def _tokenizer(self, text, token_pattern=r"(?u)\b\w\w+\b"):
        token_pattern = re.compile(token_pattern)
        return token_pattern.findall(text)

    def __call__(self, row):
        row.body = self.aug.augment(row.body)[0]
        return row


def sentence_jumbling(row):
    text = np.array(re.split(r"(\.\s|\n)+", row.body))
    np.random.shuffle(text)
    row.body = ". ".join(text)
    return row


def random_sentence(row):
    text = np.array(re.split("\s+", row.body))
    np.random.shuffle(text)
    row.body += " ".join(text[: np.random.randint(5, 15)])
    return row


def url_cleaner(row):
    text = row.body
    """Replace URLs with format: <url>: (domain/important-path)"""
    if not text:
        return text

    # Regex pattern to match URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

    def replace_url(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]

            # Extract meaningful path parts (first 1-2 segments)
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                # Take first 1-2 meaningful path segments
                important_path = "/".join(path_parts[:2])
                return f"<url>: ({domain}/{important_path})"
            else:
                return f"<url>: ({domain})"
        except:
            return "<url>: (unknown)"

    row.body = re.sub(url_pattern, replace_url, str(text))
    return row


def url_to_semantics(row):
    text = row.body
    if not isinstance(text, str):
        return ""

    url_pattern = r"https?://[^\s/$.?#].[^\s]*"

    def replace_str(match):
        url = match.group(0)
        semantic = ""
        url_lower = url.lower()

        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split(".")
            for part in parts:
                if part:
                    semantic += f"domain:{part} "

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [
            p for p in re.split(r"[/_.-]+", path) if p and p.isalnum()
        ]  # Split by common delimiters

        for part in path_parts:
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean:
                semantic += f"path:{part_clean}"

        return semantic.strip()

    row.body = re.sub(url_pattern, replace_str, text)
    return row


class RandomURL:
    def __init__(self, data):
        url_pattern = r"https?://[^\s/$.?#].[^\s]*"
        train_data = data.copy()
        train_data["urls"] = train_data.body.apply(lambda x: re.findall(url_pattern, x))
        self.urls_bank = list(itertools.chain.from_iterable(train_data.urls.values))

    def __call__(self, row):
        try:
            row.body += " " + " ".join(
                random.sample(self.urls_bank, random.randint(1, 5))
            )
        except:
            print(row.index)
        return row


def transileration(row):
    text = re.split(r"\s+", row.body)
    nwords = random.randint(min(3, len(text)), min(10, len(text)))
    iwords = random.sample(list(range(len(text))), nwords)

    for idx in iwords:
        if len(text[idx]) <= 10:
            try:
                text[idx] = re.sub(
                    r"[\s\d]", "", pronouncing.phones_for_word(text[idx])[0]
                ).lower()
            except:
                pass

    row.body = " ".join(text)
    return row


def rule_flip(row):
    row.rule = re.sub("(N|n)ot?\s+", "", row.rule)
    if "rule_violation" in row:
        row.rule_violation = int(1 - row.rule_violation)
    if "flip" in row:
        row.flip = 1
    return row
