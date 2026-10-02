"""
naive_bayes.py
--------------
A Naive Bayes text classifier implemented from scratch (no sklearn / ML
libraries) — this is the probabilistic-reasoning core of the mini project.

Concepts demonstrated (CO5):
  - Bayesian inference: P(class | message) is estimated via Bayes' rule,
    using word-frequency statistics learned from training data as the
    likelihood, combined with the class prior.
  - Acting under uncertainty: every prediction comes with a probability,
    not just a hard label.
  - Log-likelihood ratio: used both for numerically stable scoring and as
    an explainability tool (which words pushed the decision which way).
"""

import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens, stripping punctuation/numbers."""
    return TOKEN_RE.findall(text.lower())


class NaiveBayesClassifier:
    def __init__(self):
        self.classes: list[str] = []
        self.class_priors: dict[str, float] = {}
        self.word_counts: dict[str, Counter] = {}
        self.class_totals: dict[str, int] = {}
        self.vocab: set[str] = set()

    # -- Training ------------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.classes = sorted(set(labels))
        class_doc_counts = Counter(labels)
        n_docs = len(labels)

        self.word_counts = {c: Counter() for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}
        self.vocab = set()

        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.word_counts[label].update(tokens)
            self.class_totals[label] += len(tokens)
            self.vocab.update(tokens)

        self.class_priors = {c: class_doc_counts[c] / n_docs for c in self.classes}

    # -- Inference -------------------------------------------------------
    def _log_likelihood(self, word: str, cls: str) -> float:
        """P(word | class) with Laplace (add-1) smoothing, in log space."""
        count = self.word_counts[cls][word]
        return math.log((count + 1) / (self.class_totals[cls] + len(self.vocab)))

    def class_scores(self, text: str) -> dict[str, float]:
        """Raw log-probability score for each class (unnormalized)."""
        tokens = tokenize(text)
        scores = {}
        for cls in self.classes:
            score = math.log(self.class_priors[cls])
            for word in tokens:
                score += self._log_likelihood(word, cls)
            scores[cls] = score
        return scores

    def predict_proba(self, text: str) -> dict[str, float]:
        """Normalized probabilities per class (softmax over log scores)."""
        scores = self.class_scores(text)
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total = sum(exp_scores.values())
        return {c: v / total for c, v in exp_scores.items()}

    def predict(self, text: str) -> str:
        scores = self.class_scores(text)
        return max(scores, key=scores.get)

    # -- Explainability ----------------------------------------------------
    def word_contributions(self, text: str, cls_a: str, cls_b: str) -> list[tuple[str, float]]:
        """
        For each unique word in `text`, return its log-likelihood ratio
        log P(word|cls_a) - log P(word|cls_b). Positive => favors cls_a.
        Sorted by absolute strength, most influential first.
        """
        tokens = set(tokenize(text))
        contributions = []
        for word in tokens:
            ratio = self._log_likelihood(word, cls_a) - self._log_likelihood(word, cls_b)
            contributions.append((word, ratio))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions

    def top_indicator_words(self, cls_a: str, cls_b: str, n: int = 15) -> list[tuple[str, float]]:
        """Vocabulary-wide words most indicative of cls_a over cls_b."""
        scored = [
            (word, self._log_likelihood(word, cls_a) - self._log_likelihood(word, cls_b))
            for word in self.vocab
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def word_frequencies(self, cls: str, n: int = 15) -> list[tuple[str, int]]:
        return self.word_counts[cls].most_common(n)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def train_test_split(data: list[tuple[str, str]], test_ratio: float = 0.25):
    split_idx = int(len(data) * (1 - test_ratio))
    train, test = data[:split_idx], data[split_idx:]
    return train, test


def evaluate(model: NaiveBayesClassifier, texts: list[str], labels: list[str]) -> dict:
    """Return accuracy, per-class precision/recall/F1, and a confusion matrix."""
    classes = model.classes
    confusion = {actual: {predicted: 0 for predicted in classes} for actual in classes}

    correct = 0
    for text, actual in zip(texts, labels):
        predicted = model.predict(text)
        confusion[actual][predicted] += 1
        if predicted == actual:
            correct += 1

    accuracy = correct / len(labels) if labels else 0.0

    per_class = {}
    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in classes if other != cls)
        fn = sum(confusion[cls][other] for other in classes if other != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    return {"accuracy": accuracy, "per_class": per_class, "confusion": confusion}


if __name__ == "__main__":
    from dataset import generate_dataset

    data = generate_dataset(150, 150)
    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    train, test = train_test_split(list(zip(texts, labels)))
    train_texts, train_labels = zip(*train)
    test_texts, test_labels = zip(*test)

    model = NaiveBayesClassifier()
    model.fit(list(train_texts), list(train_labels))

    results = evaluate(model, list(test_texts), list(test_labels))
    print("Accuracy:", round(results["accuracy"], 3))
    for cls, m in results["per_class"].items():
        print(cls, m)
    print("Confusion matrix:", results["confusion"])

    sample = "Congratulations you have won a free vacation click here now"
    print("\nSample:", sample)
    print("Predicted:", model.predict(sample))
    print("Probabilities:", model.predict_proba(sample))
    print("Top contributing words:", model.word_contributions(sample, "spam", "ham")[:5])
