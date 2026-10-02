"""
app.py
------
Streamlit dashboard for the AI-Based Spam Classifier
(Naive Bayes / Bayesian probabilistic reasoning, built from scratch).

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import streamlit as st

from dataset import generate_dataset
from naive_bayes import NaiveBayesClassifier, evaluate, train_test_split

st.set_page_config(page_title="AI Spam Classifier Dashboard", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
# Train the model once per session (cached so it doesn't retrain on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_trained_model():
    data = generate_dataset(150, 150)
    train, test = train_test_split(data)
    train_texts, train_labels = zip(*train)
    test_texts, test_labels = zip(*test)

    model = NaiveBayesClassifier()
    model.fit(list(train_texts), list(train_labels))

    results = evaluate(model, list(test_texts), list(test_labels))
    return model, results, list(train_texts), list(train_labels)


model, eval_results, train_texts, train_labels = get_trained_model()

st.title("🛡️ AI-Based Spam Message Classifier")
st.caption("Naive Bayes probabilistic reasoning, built from scratch — Laplace smoothing, log-likelihoods, Bayesian inference.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset Overview", "🔎 Try It Live", "📈 Model Performance", "🧠 How It Reasons"]
)

# ---------------------------------------------------------------------------
# TAB 1: Dataset Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Training Data at a Glance")

    n_spam = train_labels.count("spam")
    n_ham = train_labels.count("ham")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training messages", len(train_labels))
    c2.metric("Spam examples", n_spam)
    c3.metric("Ham (normal) examples", n_ham)

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("**Most frequent words in SPAM messages**")
        words, counts = zip(*model.word_frequencies("spam", 12))
        fig, ax = plt.subplots()
        ax.barh(words[::-1], counts[::-1], color="#d9534f")
        ax.set_xlabel("Count")
        st.pyplot(fig)

    with col_right:
        st.write("**Most frequent words in HAM (normal) messages**")
        words, counts = zip(*model.word_frequencies("ham", 12))
        fig, ax = plt.subplots()
        ax.barh(words[::-1], counts[::-1], color="#5cb85c")
        ax.set_xlabel("Count")
        st.pyplot(fig)

    st.info(
        "The dataset is synthetically generated from realistic message templates "
        "(no external data needed), giving a balanced, reproducible set of spam "
        "and normal messages for the classifier to learn from."
    )

# ---------------------------------------------------------------------------
# TAB 2: Try It Live
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Classify Your Own Message")
    st.write("Type (or paste) any message and see how the AI reasons about it.")

    user_text = st.text_area(
        "Message",
        value="Congratulations! You have won a free prize, click here to claim now.",
        height=100,
    )

    if st.button("🔍 Classify"):
        if not user_text.strip():
            st.warning("Please enter a message.")
        else:
            prediction = model.predict(user_text)
            probs = model.predict_proba(user_text)

            label_display = "🚨 SPAM" if prediction == "spam" else "✅ NOT SPAM (Ham)"
            st.markdown(f"### Prediction: {label_display}")

            fig, ax = plt.subplots(figsize=(4, 1.2))
            classes = list(probs.keys())
            values = [probs[c] for c in classes]
            colors = ["#d9534f" if c == "spam" else "#5cb85c" for c in classes]
            ax.barh(classes, values, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            st.write("**Words that most influenced this decision:**")
            contributions = model.word_contributions(user_text, "spam", "ham")[:8]
            for word, ratio in contributions:
                direction = "→ pushes toward SPAM" if ratio > 0 else "→ pushes toward HAM"
                st.write(f"- `{word}` (log-likelihood ratio: {ratio:+.2f}) {direction}")

# ---------------------------------------------------------------------------
# TAB 3: Model Performance
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Evaluation on Held-Out Test Data")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{eval_results['accuracy']*100:.1f}%")
    c2.metric("Spam Precision", f"{eval_results['per_class']['spam']['precision']*100:.1f}%")
    c3.metric("Spam Recall", f"{eval_results['per_class']['spam']['recall']*100:.1f}%")
    c4.metric("Spam F1-score", f"{eval_results['per_class']['spam']['f1']*100:.1f}%")

    st.write("**Confusion Matrix**")
    classes = model.classes
    matrix = [[eval_results["confusion"][actual][pred] for pred in classes] for actual in classes]

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")
    fig.colorbar(im)
    st.pyplot(fig)

    st.write("**Per-class metrics**")
    st.table(
        {
            cls: {
                "Precision": f"{m['precision']*100:.1f}%",
                "Recall": f"{m['recall']*100:.1f}%",
                "F1-score": f"{m['f1']*100:.1f}%",
            }
            for cls, m in eval_results["per_class"].items()
        }
    )

# ---------------------------------------------------------------------------
# TAB 4: How It Reasons
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Explainability: What the Model Learned")
    st.write(
        "For each word, the model compares how likely it is to appear in spam "
        "vs. ham messages (log-likelihood ratio). Strongly positive = spam "
        "indicator, strongly negative = ham indicator."
    )

    top_spam_words = model.top_indicator_words("spam", "ham", n=12)
    top_ham_words = model.top_indicator_words("ham", "spam", n=12)

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("**Top SPAM-indicator words**")
        words = [w for w, _ in top_spam_words]
        ratios = [r for _, r in top_spam_words]
        fig, ax = plt.subplots()
        ax.barh(words[::-1], ratios[::-1], color="#d9534f")
        ax.set_xlabel("Log-likelihood ratio (favors spam)")
        st.pyplot(fig)

    with col_right:
        st.write("**Top HAM-indicator words**")
        words = [w for w, _ in top_ham_words]
        ratios = [r for _, r in top_ham_words]
        fig, ax = plt.subplots()
        ax.barh(words[::-1], ratios[::-1], color="#5cb85c")
        ax.set_xlabel("Log-likelihood ratio (favors ham)")
        st.pyplot(fig)

    with st.expander("📐 The math behind it (for your report)"):
        st.latex(r"P(\text{class} \mid \text{message}) \propto P(\text{class}) \prod_{i} P(\text{word}_i \mid \text{class})")
        st.write(
            "Each word's probability given a class is estimated from training "
            "counts using Laplace (add-1) smoothing so unseen words never get "
            "a zero probability. Scores are computed in log-space for numerical "
            "stability, then converted back to normalized probabilities for display."
        )
