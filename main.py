from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Jeet Bafna's - Documentation Helper Bot")


prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if prompt:
    with st.spinner("Generating Response.."):
        generated_response = run_llm(query=prompt)