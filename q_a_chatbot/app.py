# q_a_chatbot/app.py
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# Prompt Template
# -------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly, politely, and informatively."),
        ("user", "Question: {question}")
    ]
)

# -------------------------------
# Hugging Face LLM Creation (Cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def create_llm(model_name: str, temperature: float = 0.7, max_new_tokens: int = 150):
    """
    Create a Hugging Face pipeline and wrap it in a LangChain LLM.
    Uses CPU-friendly text-generation for Streamlit Cloud.
    """
    pipe = pipeline(
        task="text-generation",
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=-1  # Force CPU
    )
    return HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# LangChain Chain Creation (Cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def create_chain(model_name: str, temperature: float, max_tokens: int):
    llm = create_llm(model_name, temperature, max_tokens)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

# -------------------------------
# Generate Response Function
# -------------------------------
def get_response(chain, question: str) -> str:
    """
    Generate chatbot response using LangChain pipeline.
    """
    return chain.invoke({'question': question})

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hugging Face QA Chatbot", page_icon="🤖")
st.title("🤖 QA Chatbot with Hugging Face")
st.markdown("Ask any question below and get a helpful response from a Hugging Face model.")

# Sidebar settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select a model",
    ["google/flan-t5-small"]  # Only small model for fast load
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Initialize LangChain chain (cached)
chain = create_chain(model_name, temperature, max_tokens)

# User input
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        response = get_response(chain, user_input)
        st.markdown(f"**Bot:** {response}")
    except Exception as e:
        st.error(f"⚠️ Error generating response: {str(e)}")