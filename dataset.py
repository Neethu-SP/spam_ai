"""
dataset.py
----------
Builds a labeled (spam / ham) message dataset for the classifier to train on.

No internet access is required or used: messages are generated from a set of
realistic templates with randomized fill-in words, giving a dataset that's
large and varied enough for Naive Bayes to learn meaningful word-frequency
differences between spam and normal ("ham") messages, while staying fully
reproducible (fixed random seed).
"""

import random

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks for SPAM messages
# ---------------------------------------------------------------------------
SPAM_TEMPLATES = [
    "Congratulations you have won a {prize} click here to claim now",
    "URGENT your account has been suspended verify immediately or lose access",
    "Free entry in our weekly contest text WIN to claim your {prize}",
    "You have been selected for a cash reward of {amount} claim before it expires",
    "Limited time offer buy now and get {percent} percent discount instantly",
    "Your loan of {amount} has been approved click the link to receive money today",
    "Act now this exclusive {prize} is reserved only for you for 24 hours",
    "Dear customer your card has been charged {amount} click to dispute now",
    "You are the lucky winner of a {prize} reply now to collect your prize",
    "Final notice your subscription will be cancelled unless you verify now",
    "Get rich quick invest {amount} today and double your money in a week",
    "Click this link to unlock your free {prize} before it expires tonight",
    "Warning suspicious login detected verify your password immediately",
    "Exclusive deal just for you claim your {prize} now limited stock available",
    "Your parcel could not be delivered pay a small fee to reschedule delivery",
    "Hot singles in your area are waiting click now to chat for free",
    "You have unclaimed {amount} waiting text CLAIM to receive it instantly",
    "Amazing offer earn {amount} per day working from home no experience needed",
    "Your refund of {amount} is pending click to confirm your bank details",
    "Congratulations you qualify for a free {prize} respond immediately to claim",
]
SPAM_PRIZES = ["iPhone", "gift card", "vacation package", "laptop", "cash prize", "smartwatch", "voucher"]
SPAM_AMOUNTS = ["$500", "$1000", "$250", "$10000", "$75", "$2500"]
SPAM_PERCENTS = ["50", "70", "90", "30", "60"]

# ---------------------------------------------------------------------------
# Building blocks for HAM (normal) messages
# ---------------------------------------------------------------------------
HAM_TEMPLATES = [
    "Hey are we still meeting for lunch {time}",
    "Can you send me the notes from {subject} class",
    "Happy birthday hope you have a great day",
    "I will be home late tonight don't wait for dinner",
    "The assignment deadline has been extended to {time}",
    "Let's catch up this weekend if you are free",
    "Don't forget to bring the charger when you come over",
    "Thanks for helping me with the {subject} project yesterday",
    "What time does the {subject} lecture start tomorrow",
    "Can we reschedule our call to {time} instead",
    "I picked up groceries on the way back from college",
    "Mom said dinner will be ready by {time}",
    "Did you finish the {subject} assignment for tomorrow",
    "Let's meet at the library to study for the {subject} exam",
    "The bus was late again so I missed the first class",
    "Can you share the slides from today's {subject} session",
    "I'm running a bit late will reach in ten minutes",
    "Great job on the presentation today it went really well",
    "Are you free to play cricket this evening",
    "Remember to submit the {subject} report before {time}",
]
HAM_TIMES = ["tomorrow", "next week", "8 pm", "Monday", "this evening", "6 pm", "Friday"]
HAM_SUBJECTS = ["math", "AI", "history", "chemistry", "the mini project", "database", "physics"]


def _fill(template: str) -> str:
    return template.format(
        prize=random.choice(SPAM_PRIZES),
        amount=random.choice(SPAM_AMOUNTS),
        percent=random.choice(SPAM_PERCENTS),
        time=random.choice(HAM_TIMES),
        subject=random.choice(HAM_SUBJECTS),
    )


def generate_dataset(n_spam: int = 150, n_ham: int = 150) -> list[tuple[str, str]]:
    """Return a shuffled list of (message, label) pairs, label in {'spam', 'ham'}."""
    data = []
    for _ in range(n_spam):
        template = random.choice(SPAM_TEMPLATES)
        data.append((_fill(template), "spam"))
    for _ in range(n_ham):
        template = random.choice(HAM_TEMPLATES)
        data.append((_fill(template), "ham"))
    random.shuffle(data)
    return data


if __name__ == "__main__":
    for msg, label in generate_dataset(5, 5):
        print(f"[{label}] {msg}")
