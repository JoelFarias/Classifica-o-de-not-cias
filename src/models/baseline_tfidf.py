import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train_processed.csv"
VAL_PATH = PROCESSED_DIR / "validation_processed.csv"
TEST_PATH = PROCESSED_DIR / "test_processed.csv"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_split(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    return df["clean_text"], df["label"]


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )


def evaluate(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    print(f"\n=== {split_name} ===")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))
    print("Matriz de confusão:")
    print(cm)
    return report


def save_pkl(obj, path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Salvo: {path}")


def main() -> None:
    print("Carregando dados...")
    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)
    X_test, y_test = load_split(TEST_PATH)

    print("Vetorizando com TF-IDF...")
    vec = build_vectorizer()
    X_train_tfidf = vec.fit_transform(X_train)
    X_val_tfidf = vec.transform(X_val)
    X_test_tfidf = vec.transform(X_test)
    save_pkl(vec, MODELS_DIR / "tfidf_vectorizer.pkl")

    results = {}

    print("\nTreinando Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    logreg.fit(X_train_tfidf, y_train)
    save_pkl(logreg, MODELS_DIR / "baseline_logreg.pkl")
    results["logreg_val"] = evaluate(logreg, X_val_tfidf, y_val, "Logistic Regression — Validação")
    results["logreg_test"] = evaluate(logreg, X_test_tfidf, y_test, "Logistic Regression — Teste")

    print("\nTreinando LinearSVC...")
    svm = LinearSVC(C=1.0, max_iter=2000)
    svm.fit(X_train_tfidf, y_train)
    save_pkl(svm, MODELS_DIR / "baseline_svm.pkl")
    results["svm_val"] = evaluate(svm, X_val_tfidf, y_val, "LinearSVC — Validação")
    results["svm_test"] = evaluate(svm, X_test_tfidf, y_test, "LinearSVC — Teste")

    rows = []
    for model_name, report in results.items():
        for class_or_avg, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append({
                    "modelo": model_name,
                    "classe": class_or_avg,
                    "precision": round(metrics.get("precision", 0), 4),
                    "recall": round(metrics.get("recall", 0), 4),
                    "f1-score": round(metrics.get("f1-score", 0), 4),
                    "support": metrics.get("support", ""),
                })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_path = REPORTS_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados salvos em {results_path}")


if __name__ == "__main__":
    main()
