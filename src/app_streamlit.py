import sys
from pathlib import Path

import pandas as pd
import torch
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from models.deep_learning_gru import GRUClassifier, MODEL_PATH, encode_text


RESULTS_PATH = PROJECT_ROOT / "reports" / "deep_learning_results.csv"
CONFUSION_PATH = PROJECT_ROOT / "reports" / "deep_learning_confusion_matrix.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "deep_learning_run_summary.json"

EXAMPLES = {
    "World": "United Nations officials met in Geneva to discuss a new peace agreement after rising tensions between neighboring countries.",
    "Sports": "The national team won the championship after a dramatic final match, with the striker scoring twice in the second half.",
    "Business": "Shares of the company rose after investors reacted positively to quarterly earnings and stronger revenue forecasts.",
    "Sci/Tech": "Researchers announced a new artificial intelligence system that can help scientists analyze satellite images faster.",
}


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = GRUClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["vocab"], checkpoint["class_names"], config


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_confusion_matrix():
    if not CONFUSION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CONFUSION_PATH, index_col=0)


def predict(text: str):
    model, vocab, class_names, config = load_model()
    input_ids = torch.tensor([encode_text(text, vocab, config["max_length"])], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        prediction = int(torch.argmax(probabilities).item())

    probability_table = pd.DataFrame(
        {
            "Classe": class_names,
            "Probabilidade": [float(value) for value in probabilities],
        }
    ).sort_values("Probabilidade", ascending=False)

    return class_names[prediction], float(probabilities[prediction]), probability_table


def confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "Alta"
    if confidence >= 0.50:
        return "Media"
    return "Baixa"


def render_sidebar():
    _, _, _, config = load_model()
    results = load_results()
    test_accuracy = None

    if not results.empty:
        accuracy_row = results[(results["modelo"] == "gru_test") & (results["classe"] == "accuracy")]
        if not accuracy_row.empty:
            test_accuracy = float(accuracy_row.iloc[0]["f1-score"])

    st.sidebar.header("Modelo")
    st.sidebar.write("GRU bidirecional")
    st.sidebar.metric("Acuracia no teste", f"{test_accuracy:.2%}" if test_accuracy else "N/A")
    st.sidebar.metric("Epocas", config["epochs"])
    st.sidebar.metric("Amostras de treino", f"{config['max_train_samples']:,}".replace(",", "."))
    st.sidebar.metric("Vocabulario", f"{config['vocab_size']:,}".replace(",", "."))


def render_classifier():
    st.subheader("Classificar noticia")

    selected_example = st.selectbox("Exemplo rapido", ["Texto livre"] + list(EXAMPLES.keys()))
    default_text = "" if selected_example == "Texto livre" else EXAMPLES[selected_example]

    news_text = st.text_area(
        "Noticia",
        value=default_text,
        height=180,
        placeholder="Cole aqui o titulo e a descricao de uma noticia.",
    )

    classify = st.button("Classificar", type="primary", use_container_width=True)

    if classify:
        if not news_text.strip():
            st.error("Digite ou cole uma noticia.")
            return

        label, confidence, probability_table = predict(news_text)

        col1, col2, col3 = st.columns(3)
        col1.metric("Classe prevista", label)
        col2.metric("Confianca", f"{confidence:.2%}")
        col3.metric("Nivel", confidence_label(confidence))

        st.progress(min(max(confidence, 0.0), 1.0))

        chart_data = probability_table.set_index("Classe")
        st.bar_chart(chart_data)

        st.dataframe(
            probability_table.assign(Probabilidade=probability_table["Probabilidade"].map(lambda value: f"{value:.2%}")),
            hide_index=True,
            use_container_width=True,
        )


def render_metrics():
    st.subheader("Metricas do modelo")
    results = load_results()
    if results.empty:
        st.info("Arquivo de metricas nao encontrado.")
        return

    test_rows = results[results["modelo"] == "gru_test"].copy()
    class_rows = test_rows[test_rows["classe"].isin(["World", "Sports", "Business", "Sci/Tech"])]
    macro_row = test_rows[test_rows["classe"] == "macro avg"]

    if not macro_row.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision media", f"{float(macro_row.iloc[0]['precision']):.2%}")
        col2.metric("Recall medio", f"{float(macro_row.iloc[0]['recall']):.2%}")
        col3.metric("F1 medio", f"{float(macro_row.iloc[0]['f1-score']):.2%}")

    st.dataframe(class_rows, hide_index=True, use_container_width=True)


def render_confusion_matrix():
    st.subheader("Matriz de confusao")
    matrix = load_confusion_matrix()
    if matrix.empty:
        st.info("Matriz de confusao nao encontrada.")
        return

    st.dataframe(matrix, use_container_width=True)

    figure_path = PROJECT_ROOT / "reports" / "figures" / "deep_learning_confusion_matrix.png"
    if figure_path.exists():
        st.image(str(figure_path), use_column_width=True)


st.set_page_config(page_title="Classificador de Noticias", layout="wide")
st.title("Classificador de Noticias")

if not MODEL_PATH.exists():
    st.warning("Modelo nao encontrado. Rode primeiro: python src/models/deep_learning_gru.py")
    st.stop()

render_sidebar()

tab_classifier, tab_metrics, tab_confusion = st.tabs(["Classificador", "Metricas", "Matriz de confusao"])

with tab_classifier:
    render_classifier()

with tab_metrics:
    render_metrics()

with tab_confusion:
    render_confusion_matrix()
