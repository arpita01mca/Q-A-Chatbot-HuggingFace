import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

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
    ["google/flan-t5-small", "google/flan-t5-base"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# -------------------------------
# Load LLM
# -------------------------------
@st.cache_resource
def create_llm(model_name, temperature=0.7, max_new_tokens=150):
    """
    Create a Hugging Face text-generation pipeline and wrap it with LangChain.
    """
    # Create HF pipeline
    pipe = pipeline(
        task="text-generation",  # universal, works for Flan-T5
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    # Wrap with LangChain
    return HuggingFacePipeline(pipeline=pipe)

# Initialize LLM
llm = create_llm(model_name, temperature, max_tokens)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        # Construct prompt with system instruction
        prompt_text = f"You are a helpful assistant. Answer the following question clearly and politely:\n{user_input}"
        # Generate response
        output = llm(prompt_text)
        # HuggingFacePipeline returns a string, so we can display directly
        st.markdown(f"**Bot:** {output}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
