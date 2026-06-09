import sys
from pathlib import Path

import pandas as pd
import torch
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from models.deep_learning_gru import GRUClassifier, MODEL_PATH, encode_text


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
    return model, checkpoint["vocab"], checkpoint["class_names"], config["max_length"]


def predict(text: str):
    model, vocab, class_names, max_length = load_model()
    input_ids = torch.tensor([encode_text(text, vocab, max_length)], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        prediction = int(torch.argmax(probabilities).item())

    return class_names[prediction], float(probabilities[prediction]), probabilities.tolist(), class_names


st.set_page_config(page_title="Classificador de Noticias")
st.title("Classificador de Noticias")

if not MODEL_PATH.exists():
    st.warning("Modelo nao encontrado. Rode primeiro: python src/models/deep_learning_gru.py")
    st.stop()

news_text = st.text_area("Texto da noticia", height=180)

if st.button("Classificar"):
    if not news_text.strip():
        st.error("Digite ou cole uma noticia.")
    else:
        label, confidence, probabilities, class_names = predict(news_text)
        st.subheader(label)
        st.write(f"Confianca: {confidence:.2%}")
        st.bar_chart(pd.Series(probabilities, index=class_names))
