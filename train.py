# train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def create_sample_dataset(csv_path: str):
    """Generates a small starter dataset across 5 core complaint categories."""
    data = [
        # Billing
        ("I was charged twice on my credit card for the same monthly subscription.", "Billing"),
        ("My monthly invoice shows an unexpected hidden fee that I never agreed to.", "Billing"),
        ("Please issue a refund immediately; the money was deducted without authorization.", "Billing"),
        ("Overcharged on my latest bill, need an adjusted statement.", "Billing"),
        ("Why did my billing amount increase this month without prior notice?", "Billing"),

        # Technical Issues
        ("The mobile app keeps crashing every time I reach the checkout screen.", "Technical Issues"),
        ("I keep seeing a 500 internal server error whenever I try to upload my document.", "Technical Issues"),
        ("The website dashboard is down, completely unresponsive on Chrome.", "Technical Issues"),
        ("The API connection keeps timing out after sending the request.", "Technical Issues"),
        ("Software freezes on startup; I have reinstalled it three times.", "Technical Issues"),

        # Product Quality
        ("The laptop keyboard arrived damaged and several keys are unresponsive.", "Product Quality"),
        ("The stitching on the jacket ripped apart after wearing it just twice.", "Product Quality"),
        ("The packaging was completely crushed and the screen inside was scratched.", "Product Quality"),
        ("The item feels cheap and does not match the specifications displayed online.", "Product Quality"),
        ("Battery health dropped to 60 percent after only one week of regular use.", "Product Quality"),

        # Service Delivery
        ("My package has been delayed for two weeks with no tracking update available.", "Service Delivery"),
        ("The delivery driver marked my order as handed to recipient, but nothing arrived.", "Service Delivery"),
        ("Scheduled an installation appointment for today, but no technician showed up.", "Service Delivery"),
        ("Courier sent my shipment to the completely wrong delivery hub in another state.", "Service Delivery"),
        ("Estimated arrival was Monday; it is now Friday and support provides no update.", "Service Delivery"),

        # Account Security
        ("I received an email stating an unauthorized login attempt from an unknown IP.", "Account Security"),
        ("My account was hacked and my registered phone number was altered.", "Account Security"),
        ("Potential fraud detected: two-factor authentication codes are arriving without my action.", "Account Security"),
        ("Someone breached my profile and changed the recovery email address.", "Account Security"),
        ("I am locked out of my portal because a suspicious login reset my password.", "Account Security")
    ]
    df = pd.DataFrame(data, columns=["complaint_text", "category"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Generated sample records at {csv_path}")
    return df

def train_and_export():
    dataset_path = os.path.join("data", "complaints.csv")
    model_dir = "models"
    model_path = os.path.join(model_dir, "complaint_model.joblib")

    # 1. Load or synthesize data
    if not os.path.exists(dataset_path):
        df = create_sample_dataset(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    X = df["complaint_text"]
    y = df["category"]

    # 2. Stratified split to ensure equal representation in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Build unified pipeline: TF-IDF feature extractor + Naive Bayes classifier
    # stop_words='english' removes common non-informative words (e.g., 'the', 'is')
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    # 4. Fit model
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    predictions = pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\n[Evaluation] Baseline Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))

    # 6. Save model artifact
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[Export] Trained pipeline saved to {model_path}")

if __name__ == "__main__":
    train_and_export()