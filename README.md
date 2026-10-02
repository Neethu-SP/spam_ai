# AI-Based Spam Message Classifier — Probabilistic Reasoning Dashboard

Mini project for 23AD36C — Artificial Intelligence (B.Tech AI&DS, NEC).
Maps to **CO5 (probabilistic reasoning / Bayesian inference)** with
supporting elements of CO2 (heuristic feature use) and CO11 (evaluation).

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the dashboard:
   ```
   streamlit run app.py
   ```
   (If `streamlit` isn't recognized as a command, use `python -m streamlit run app.py` instead.)
3. Your browser opens automatically at `http://localhost:8501`.

## Files

| File              | Purpose                                                              |
|-------------------|-----------------------------------------------------------------------|
| `dataset.py`      | Generates a synthetic, labeled spam/ham message dataset from templates. |
| `naive_bayes.py`  | The AI core — Naive Bayes classifier built from scratch (no sklearn), plus evaluation metrics. |
| `app.py`          | The 4-tab Streamlit dashboard.                                       |

## How the AI works

1. **Bayesian classification.** For a message to be spam or ham, the model
   estimates `P(class | message)` using Bayes' rule: a class prior times the
   product of each word's likelihood under that class, computed in log-space
   for numerical stability.
2. **Laplace smoothing.** Word probabilities are smoothed (add-1) so a word
   never seen in training doesn't wrongly force a probability of zero.
3. **Explainability.** For any message, the log-likelihood ratio of each
   word between the two classes shows exactly which words pushed the
   decision toward spam or ham — this powers the "How It Reasons" tab.
4. **Evaluation.** The dataset is split into train/test sets; accuracy,
   precision, recall, F1-score, and a confusion matrix are computed on the
   held-out test set (Tab 3).

## Dashboard tabs

1. **📊 Dataset Overview** — class balance + most frequent words per class.
2. **🔎 Try It Live** — type any message, get a prediction with a probability
   chart and the specific words that drove the decision.
3. **📈 Model Performance** — accuracy/precision/recall/F1 + confusion matrix heatmap.
4. **🧠 How It Reasons** — top spam-indicator and ham-indicator words
   (ranked by log-likelihood ratio), plus the underlying formula.

## Suggested report content

- Explain Bayes' theorem and how it's applied here (Tab 4 has the formula).
- Include the confusion matrix and metrics from Tab 3 as evidence of
  evaluation methodology.
- Discuss Laplace smoothing and why it's necessary (handles unseen words).
- Note the dataset is synthetically generated via templates for full
  reproducibility and balanced class representation — mention this as your
  data collection methodology.
