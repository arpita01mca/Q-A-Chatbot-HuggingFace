import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chains import LLMChain

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
    Create a Hugging Face text-generation pipeline and wrap it as a LangChain LLM.
    """
    pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = create_llm(model_name, temperature, max_tokens)

# -------------------------------
# Prompt Template & Chain
# -------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer questions clearly, politely, and informatively."),
        ("user", "Question: {question}")
    ]
)
output_parser = StrOutputParser()
chain = LLMChain(prompt=prompt, llm=llm, output_parser=output_parser)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        response = chain.invoke({"question": user_input})
        st.markdown(f"**Bot:** {response}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
