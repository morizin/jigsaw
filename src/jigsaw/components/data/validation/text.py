from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud, STOPWORDS
from ....core import Directory
from typeguard import typechecked
import matplotlib.pyplot as plt
from pandas import Series
import seaborn as sns
import pandas as pd
import numpy as np
import statistics


@typechecked
def get_statistics(
    data: list[str] | Series, column: str, path: Directory
) -> dict[str, str | dict[str, int | float]]:
    stat: dict[str, str | dict[str, int | float]] = {"type": "text"}
    if isinstance(data, Series):
        data = data.tolist()
    length = list(map(lambda x: len(x.split()), data))

    quantiles = statistics.quantiles(length)
    stat["word_length"] = {
        "sum": sum(length),
        "min": min(length),
        "max": max(length),
        "mean": round(statistics.mean(length), 4),
        "median": round(statistics.median(length), 4),
        "mode": round(statistics.mode(length), 4),
        "variance": round(statistics.variance(length), 4),
        "stdev": round(statistics.stdev(length), 4),
        "1%": round(statistics.quantiles(length, n=100)[0], 4),
        "25%": round(quantiles[0], 4),
        "50%": round(quantiles[1], 4),
        "75%": round(quantiles[2], 4),
        "99%": round(statistics.quantiles(length, n=100)[-1], 4),
        "histogram": f"{column}_word_length.png",
    }

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.kdeplot(data=length, fill=True, ax=ax2)
    ax2.set_title("Word Length (KDE)")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    fig2.savefig(path // "light" / stat["word_length"]["histogram"])

    with plt.style.context("dark_background"):
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sns.kdeplot(data=length, fill=True, ax=ax1)
        ax1.set_title("Word Length (KDE)")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Density")
        fig1.savefig(path // "dark" / stat["word_length"]["histogram"])

    length = list(map(len, data))
    stat["char_length"] = {
        "sum": sum(length),
        "min": min(length),
        "max": max(length),
        "mean": round(statistics.mean(length), 4),
        "median": round(statistics.median(length), 4),
        "mode": round(statistics.mode(length), 4),
        "variance": round(statistics.variance(length), 4),
        "stdev": round(statistics.stdev(length), 4),
        "1%": round(statistics.quantiles(length, n=100)[0], 4),
        "25%": round(quantiles[0], 4),
        "50%": round(quantiles[1], 4),
        "75%": round(quantiles[2], 4),
        "99%": round(statistics.quantiles(length, n=100)[-1], 4),
        "histogram": f"{column}_char_length.png",
    }
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.kdeplot(data=length, fill=True, ax=ax3)
    ax3.set_title("Char Length (KDE)")
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Density")
    fig3.savefig(path // "light" / stat["char_length"]["histogram"])

    with plt.style.context("dark_background"):
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.kdeplot(data=length, fill=True, ax=ax4)
        ax4.set_title("Char Length (KDE)")
        ax4.set_xlabel("Value")
        ax4.set_ylabel("Density")
        fig4.savefig(path // "dark" / stat["char_length"]["histogram"])

    # Now we need unique vocabs in column
    words: list[str] = []
    for sentence in data:
        words += sentence.split()

    stat["total_vocabs"] = len(words)
    stat["total_unique_vocabs"] = len(set(words))

    stat["uniqueness"] = round(
        100 * stat["total_unique_vocabs"] / stat["total_vocabs"], 4
    )

    return stat


@typechecked
def generate_word_cloud(
    data: pd.DataFrame, column: str, path: Directory, seed: int = 1234
) -> str:
    data: Series = data[column]
    _ = (
        WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=set(STOPWORDS),
            random_state=seed,
        )
        .generate(" ".join(data))
        .to_file(path // "light" / f"{column}_word_cloud.png")
    )

    _ = (
        WordCloud(
            width=800,
            height=400,
            background_color="black",
            stopwords=set(STOPWORDS),
            random_state=seed,
        )
        .generate(" ".join(data))
        .to_file(path // "dark" / f"{column}_word_cloud.png")
    )
    return f"{column}_word_cloud.png"


@typechecked
def detect_data_drift(
    data: pd.DataFrame,
    column: str,
    path: Directory,
    current_data: pd.DataFrame | None = None,
    n_splits: int = 5,
    n_iteration: int = 100,
    dimension: int = 500,
    seed: int = 1234,
) -> dict:
    stat = {}
    model = TfidfVectorizer(max_features=dimension, ngram_range=(1, 2))
    if current_data is None:
        data["token_length"] = data[column].apply(lambda x: len(x.split()))
        data = data.sort_values(by="token_length", ignore_index=True)
        data["fold"] = data.index % n_splits
        embeddings = model.fit_transform(data[column])
        train_embeddings, test_embeddings, _, _ = train_test_split(
            embeddings,
            data["fold"],
            test_size=1 / n_splits,
            random_state=seed,
            stratify=data["fold"],
        )
    else:
        data["fold"] = 0
        current_data["fold"] = 1
        data = pd.concat(
            [data[[column, "fold"]], current_data[[column, "fold"]]], axis=0
        )
        embeddings = model.fit_transform(data[column])
        train_embeddings = model.transform(data[data["fold"] == 0][column])
        test_embeddings = model.transform(data[data["fold"] == 1][column])
        n_splits = 2

    train_mean = train_embeddings.mean(axis=0)
    train_mean /= np.linalg.norm(train_mean)
    test_mean = test_embeddings.mean(axis=0)
    test_mean /= np.linalg.norm(test_mean)
    stat["euclidean_mean_distance"] = round(np.linalg.norm(train_mean - test_mean), 4)
    stat["cosine_mean_similarity"] = round((train_mean @ test_mean.T)[0, 0], 4)
    pca_model = PCA(n_components=2, random_state=seed)
    train_embeddings_2d = pca_model.fit_transform(train_embeddings)
    embeddings2d = pca_model.transform(embeddings)
    test_embeddings_2d = pca_model.transform(test_embeddings)

    stat["scatter_embedding"] = f"{column}_scatter_embedding.png"
    stat["embedding_histogram"] = f"{column}_embedding_histogram.png"

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(
        embeddings2d[:, 0],
        embeddings2d[:, 1],
        c=data["fold"],
    )
    ax1.set_title("Embeddings Scatter Plot (Light)")
    fig1.savefig(path // "light" / stat["scatter_embedding"])
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 5), ncols=2)
    for dim in range(embeddings2d.shape[1]):
        sns.kdeplot(data=train_embeddings_2d[:, dim], fill=False, ax=ax2[dim])
        sns.kdeplot(data=test_embeddings_2d[:, dim], fill=False, ax=ax2[dim])
        ax2[dim].set_title(f"Embedding Histogram Dim {dim} (KDE) (Light)")
        ax2[dim].set_xlabel("Value")
        ax2[dim].set_ylabel("Density")
    fig2.savefig(path // "light" / stat["embedding_histogram"])
    plt.close(fig2)

    with plt.style.context("dark_background"):
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.scatter(
            embeddings2d[:, 0],
            embeddings2d[:, 1],
            c=data["fold"],
        )
        ax3.set_title("Embeddings Scatter Plot (Dark)")
        fig3.savefig(path // "dark" / stat["scatter_embedding"])
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(14, 5), ncols=2)
        for dim in range(embeddings2d.shape[1]):
            sns.kdeplot(data=train_embeddings_2d[:, dim], fill=False, ax=ax4[dim])
            sns.kdeplot(data=test_embeddings_2d[:, dim], fill=False, ax=ax4[dim])

            ax4[dim].set_title(f"Embedding Histogram Dim {dim} (KDE) (Dark)")
            ax4[dim].set_xlabel("Value")
            ax4[dim].set_ylabel("Density")
        fig4.savefig(path // "dark" / stat["embedding_histogram"])
        plt.close(fig4)

    embeddings, tembeddings, labels, tlabels = train_test_split(
        embeddings,
        data["fold"],
        test_size=1 / n_splits,
        random_state=seed,
        stratify=data["fold"],
    )
    model = RandomForestClassifier(n_estimators=n_iteration, random_state=seed)
    model.fit(embeddings, labels)
    preds = model.predict_proba(tembeddings)
    stat["accuracy"] = round(accuracy_score(tlabels, np.argmax(preds, axis=1)), 4)
    stat["auc_score"] = round(
        roc_auc_score(
            tlabels,
            preds if current_data is None else preds[:, 1],
            multi_class="ovr",
        ),
        4,
    )

    return stat
