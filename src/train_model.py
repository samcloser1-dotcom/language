import os
import pickle
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "hand_landmarks.csv")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def train_sign_model(csv_file=CSV_PATH):
    """
    Loads dataset, preprocesses landmarks, trains Multi-Layer Perceptron neural network,
    evaluates model metrics, and persists trained artifacts.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Dataset CSV not found at '{csv_file}'. "
            "Please run 'python create_sample_data.py' or 'python src/data_collection.py' first!"
        )

    print(f"[Model Training] Loading landmark dataset from '{csv_file}'...")
    df = pd.read_csv(csv_file)

    if df.empty or len(df) < 5:
        raise ValueError("Dataset CSV is empty or has insufficient rows for training.")

    # Split features X and targets y
    y_raw = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    print(f"[Model Training] Dataset loaded with {X.shape[0]} total samples and {X.shape[1]} features.")
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = [str(c) for c in label_encoder.classes_]
    print(f"[Model Training] Identified {len(classes)} distinct gesture classes: {classes}")

    # Stratified Train-Test Split (fallback if any class has < 2 samples)
    min_class_samples = pd.Series(y).value_counts().min()
    use_stratify = y if min_class_samples >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=use_stratify
    )

    print(f"[Model Training] Training Set: {X_train.shape[0]} samples | Testing Set: {X_test.shape[0]} samples")

    # Neural Network Architecture: MLP Deep Classifier
    print("[Model Training] Initializing Multi-Layer Perceptron (MLP) Classifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=600,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
        verbose=False
    )

    # Train Neural Network
    print("[Model Training] Training neural network model...")
    model.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("                MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Detailed Classification Report (Precision / Recall / F1-Score):")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0.0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix Overview (Diagonal = Correct Predictions):")
    print(cm)
    print("=" * 60 + "\n")

    # Save Model Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"[Model Training] Model saved successfully to '{MODEL_PATH}'")
    print(f"[Model Training] Label Encoder saved to '{ENCODER_PATH}'")

    return accuracy, model, label_encoder


if __name__ == "__main__":
    train_sign_model()
