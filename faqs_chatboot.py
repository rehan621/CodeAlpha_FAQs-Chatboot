import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Load env
load_dotenv()

st.set_page_config(page_title="FAQ Chatbot", layout="centered")
st.title("🏫 School FAQ Chatbot+AI")
st.caption("Ask questions about admissions, fees, classes, teachers & facilities")



# Load NLP (SAFE)
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        return spacy.blank("en")   # fallback (NO CRASH)

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp = load_nlp()
embed_model = load_embedding_model()

llm = ChatOpenAI()


# Preprocess function (OUTSIDE cache)
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join(
        [t.lemma_ for t in doc if not (t.is_stop or t.is_punct)]
    )


# Load FAQs (ONLY DATA)
@st.cache_data
def load_faqs():
    df = pd.read_csv("data/faqs.csv").dropna()
     
    clean_questions = [preprocess(q) for q in df["question"]]
    embeddings = embed_model.encode(
        clean_questions,
        convert_to_numpy=True
    )
    return df, embeddings

df, faq_embeddings = load_faqs()


# FAQ Search
def faq_search(user_question, threshold=0.45):
    uq = preprocess(user_question)
    uq_emb = embed_model.encode([uq], convert_to_numpy=True)
    scores = cosine_similarity(uq_emb, faq_embeddings)[0]
    best_idx = np.argmax(scores)

    if scores[best_idx] >= threshold:
        return df.iloc[best_idx]["answer"]
    return None


# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
       SystemMessage(
    content="You are a helpful school information assistant. "
            "Answer questions related to school admissions, fees, subjects, "
            "teachers, rules, facilities, and academics."
)
]

for msg in st.session_state.chat_history[1:]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    faq_answer = faq_search(user_input)

    if faq_answer:
        reply = faq_answer
    else:
        response = llm.invoke(st.session_state.chat_history)
        reply = response.content

    st.session_state.chat_history.append(
        AIMessage(content=reply)
    )

    with st.chat_message("assistant"):
        st.markdown(reply)
