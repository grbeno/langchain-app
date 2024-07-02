## $ streamlit run Welcome.py
## deploy -> https://streamlit.io/cloud

import streamlit as st

st.set_page_config(
        page_title="Welcome",
        page_icon="👋",
    )

## Style
# with open('style.css') as f:
#     css = f.read()
# st.write(f'<style>{css}</style>', unsafe_allow_html=True)

## Content

st.write("# My Langchain-Streamlit App! 👋")

st.markdown(
    """
    ---
    ### Content
    - Chat with an AI (OpenAI, HuggingFace) (test version)
    - Backend interview with an AI (test version)
    - Documentation reading with an AI (planned)
"""
)

