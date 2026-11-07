import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# Configuração da página
# --------------------------------------------------
st.set_page_config(
    page_title="Analisador de Sentimentos | BERTimbau",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODELO_ID = "minion1388/bertimbau-b2w"
MAX_LENGTH = 128

# --------------------------------------------------
# Estilos
# --------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f5f3ff 0%, #fafafa 48%, #ffffff 100%);
        }
        .block-container {
            max-width: 850px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }
        .hero {
            padding: 1.7rem 1.8rem;
            border-radius: 22px;
            background: linear-gradient(120deg, #312e81 0%, #6d5bd0 58%, #8b7ce8 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 12px 30px rgba(49, 46, 129, 0.15);
        }
        .hero h1 {
            color: white;
            font-size: 2rem;
            line-height: 1.2;
            margin: 0 0 .55rem 0;
        }
        .hero p {
            color: #eeecff;
            font-size: 1rem;
            margin: 0;
        }
        .section-label {
            color: #4338a8;
            font-weight: 700;
            font-size: .9rem;
            letter-spacing: .04em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .result-positive, .result-negative {
            padding: 1.15rem 1.25rem;
            border-radius: 16px;
            margin: .5rem 0 1rem 0;
        }
        .result-positive {
            background: #ecfdf3;
            border: 1px solid #a7e8bf;
        }
        .result-negative {
            background: #fff1f1;
            border: 1px solid #ffc4c4;
        }
        .result-title {
            font-size: 1.35rem;
            font-weight: 750;
            margin: 0;
        }
        .result-positive .result-title { color: #167342; }
        .result-negative .result-title { color: #b42318; }
        .result-caption {
            color: #555;
            font-size: .9rem;
            margin-top: .25rem;
        }
        div[data-testid="stTextArea"] textarea {
            border-radius: 12px;
        }
        div.stButton > button {
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 650;
        }
        .footnote {
            color: #777;
            font-size: .82rem;
            text-align: center;
            margin-top: 1.4rem;
        }
        @media (max-width: 640px) {
            .block-container { padding: 1rem 1rem 2rem 1rem; }
            .hero { padding: 1.35rem; border-radius: 17px; }
            .hero h1 { font-size: 1.65rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Carregamento do modelo (em cache para evitar recargas)
# --------------------------------------------------
@st.cache_resource(show_spinner="Carregando o modelo BERTimbau...")
def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_ID)
    modelo.eval()
    return tokenizer, modelo


def obter_rotulo(modelo, indice):
    """Obtém o rótulo configurado no modelo e normaliza sua apresentação."""
    rotulo = modelo.config.id2label.get(indice, str(indice))
    rotulo_normalizado = str(rotulo).strip().lower()

    if "posit" in rotulo_normalizado:
        return "Positivo"
    if "negat" in rotulo_normalizado:
        return "Negativo"

    # Fallback compatível com a codificação usual do projeto: 0 = negativo; 1 = positivo.
    return "Positivo" if indice == 1 else "Negativo"


def analisar_avaliacao(texto):
    tokenizer, modelo = carregar_modelo()
    entradas = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        saida = modelo(**entradas)
        probabilidades = F.softmax(saida.logits, dim=1)[0]

    indice = int(torch.argmax(probabilidades).item())
    sentimento = obter_rotulo(modelo, indice)
    pontuacao = float(probabilidades[indice].item() * 100)
    return sentimento, pontuacao


# --------------------------------------------------
# Interface
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💬 Analisador de Sentimentos</h1>
        <p>Descubra se uma avaliação de produto expressa um sentimento positivo ou negativo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-label'>Faça sua análise</div>", unsafe_allow_html=True)
st.write("Digite ou cole uma avaliação de produto no campo abaixo.")

with st.container(border=True):
    avaliacao = st.text_area(
        "Avaliação do produto",
        placeholder="Ex.: O produto chegou antes do prazo e tem ótima qualidade. Recomendo!",
        height=160,
        max_chars=5000,
        label_visibility="collapsed",
    )
    analisar = st.button("🔎 Analisar sentimento", type="primary", use_container_width=True)

if analisar:
    if not avaliacao.strip():
        st.warning("Digite uma avaliação antes de solicitar a análise.")
        st.session_state.pop("resultado_analise", None)
    else:
        try:
            with st.spinner("Analisando o texto..."):
                sentimento, pontuacao = analisar_avaliacao(avaliacao.strip())
            st.session_state["resultado_analise"] = {
                "texto": avaliacao.strip(),
                "sentimento": sentimento,
                "pontuacao": pontuacao,
            }
        except Exception as erro:
            st.error("Não foi possível realizar a análise. Tente novamente em alguns instantes.")
            st.caption(f"Detalhe técnico: {erro}")
            st.session_state.pop("resultado_analise", None)

resultado = st.session_state.get("resultado_analise")
if resultado:
    st.markdown("### Resultado da análise")

    if resultado["sentimento"] == "Positivo":
        st.markdown(
            """
            <div class="result-positive">
                <p class="result-title">😊 Sentimento positivo</p>
                <p class="result-caption">O modelo identificou predominância de características positivas na avaliação.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-negative">
                <p class="result-title">😞 Sentimento negativo</p>
                <p class="result-caption">O modelo identificou predominância de características negativas na avaliação.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.metric("Pontuação atribuída pelo modelo", f"{resultado['pontuacao']:.2f}%")
    st.progress(min(max(resultado["pontuacao"] / 100, 0.0), 1.0))
    st.caption(
        "Essa pontuação representa a confiança calculada pelo modelo para a classe escolhida. "
        "Ela não garante que a classificação esteja correta."
    )

st.markdown(
    """
    <div class="footnote">
        Projeto acadêmico de Processamento de Linguagem Natural<br>
        Modelo: BERTimbau · Base de dados: B2W-Reviews01
    </div>
    """,
    unsafe_allow_html=True,
)
