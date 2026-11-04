import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Configuração da página
st.set_page_config(
    page_title="Analisador de Sentimentos",
    page_icon="💬",
    layout="centered"
)

# Identificação do modelo no Hugging Face
MODELO_ID = "minion1388/bertimbau-b2w"

@st.cache_resource
def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_ID)
    modelo.eval()
    return tokenizer, modelo

# Título
st.title("💬 Analisador de Sentimentos")
st.write(
    "Analise avaliações de produtos utilizando "
    "o modelo BERTimbau treinado com a base B2W Reviews."
)

st.divider()

# Campo de texto
avaliacao = st.text_area(
    "Digite sua avaliação:",
    placeholder="Exemplo: O produto chegou rápido e é de ótima qualidade!",
    height=150
)

# Botão de análise
if st.button("Analisar sentimento", type="primary"):

    if not avaliacao.strip():
        st.warning("Por favor, digite uma avaliação.")

    else:
        with st.spinner("Analisando sua avaliação..."):

            tokenizer, modelo = carregar_modelo()

            entradas = tokenizer(
                avaliacao,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            with torch.no_grad():
                saida = modelo(**entradas)
                probabilidades = F.softmax(saida.logits, dim=1)[0]

            indice = torch.argmax(probabilidades).item()

            sentimentos = modelo.config.id2label
            sentimento = sentimentos[indice]

            confianca = probabilidades[indice].item() * 100

        st.divider()
        st.subheader("Resultado da análise")

        if sentimento.lower() == "positivo":
            st.success("😊 Sentimento identificado: POSITIVO")
        else:
            st.error("😞 Sentimento identificado: NEGATIVO")

        st.metric("Confiança do modelo", f"{confianca:.2f}%")

st.divider()

st.caption("Projeto acadêmico de Processamento de Linguagem Natural.")
st.caption("Modelo: BERTimbau | Base: B2W Reviews")