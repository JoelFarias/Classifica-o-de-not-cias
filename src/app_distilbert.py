from pathlib import Path

import pandas as pd
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_news_model"
RESULTS_PATH = PROJECT_ROOT / "reports" / "distilbert_results.csv"
CONFUSION_PATH = PROJECT_ROOT / "reports" / "distilbert_confusion_matrix.csv"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
EXAMPLES = {
    "World": "United Nations officials met to discuss a peace agreement after diplomatic tensions escalated.",
    "Sports": "The football team won the final after scoring twice in the last ten minutes.",
    "Business": "The stock market rose after major companies reported stronger quarterly profits.",
    "Sci/Tech": "Scientists announced a new artificial intelligence tool for medical image analysis.",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH)


def predict(text: str):
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0)
        prediction = int(torch.argmax(probabilities).item())

    table = pd.DataFrame(
        {
            "Classe": CLASS_NAMES,
            "Probabilidade": [float(value) for value in probabilities],
        }
    ).sort_values("Probabilidade", ascending=False)

    return CLASS_NAMES[prediction], float(probabilities[prediction]), table


st.set_page_config(page_title="Classificador DistilBERT", layout="wide")
st.title("Classificador de Noticias - DistilBERT")

if not MODEL_DIR.exists():
    st.warning("Modelo DistilBERT nao encontrado. Rode primeiro: python src/models/deep_learning_distilbert.py")
    st.stop()

selected_example = st.selectbox("Exemplo rapido", ["Texto livre"] + list(EXAMPLES.keys()))
default_text = "" if selected_example == "Texto livre" else EXAMPLES[selected_example]
news_text = st.text_area("Noticia", value=default_text, height=180)

if st.button("Classificar", type="primary"):
    if not news_text.strip():
        st.error("Digite ou cole uma noticia.")
    else:
        label, confidence, probability_table = predict(news_text)
        col1, col2 = st.columns(2)
        col1.metric("Classe prevista", label)
        col2.metric("Confianca", f"{confidence:.2%}")
        st.bar_chart(probability_table.set_index("Classe"))
        st.dataframe(
            probability_table.assign(Probabilidade=probability_table["Probabilidade"].map(lambda value: f"{value:.2%}")),
            hide_index=True,
            use_container_width=True,
        )

results = load_results()
if not results.empty:
    st.subheader("Metricas")
    st.dataframe(results, hide_index=True, use_container_width=True)

if CONFUSION_PATH.exists():
    st.subheader("Matriz de confusao")
    st.dataframe(pd.read_csv(CONFUSION_PATH, index_col=0), use_container_width=True)
