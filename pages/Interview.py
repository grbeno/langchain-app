# https://python.langchain.com/v0.1/docs/integrations/memory/streamlit_chat_message_history/
# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py

import os

import streamlit as st
from streamlit_authenticator import Authenticate
from environs import Env
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
import yaml
from yaml.loader import SafeLoader

## Style
# with open('style.css') as f:
#     css = f.read()
# st.write(f'<style>{css}</style>', unsafe_allow_html=True)

## $ streamlit run app.py

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Login'})

## Environment variables

env = Env()
env.read_env()

## Langsmith

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = env.str("LANGCHAIN_API_KEY")

if authentication_status:
    
    st.sidebar.write(f'Welcome *{username}*')
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.divider()
    
    st.sidebar.success("Tech interviewer")

    model = st.sidebar.selectbox("Select model", ['gpt-4o','gpt-4','gpt-3.5-turbo'])

    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tmp = st.sidebar.select_slider(
        "Set temperature",
        options=scale,
        value=0.4,
    )

    lang = st.sidebar.radio("Language (answer)", ["English", "Magyar", "Deutsch"], horizontal=True)
    st.sidebar.divider()
    st.sidebar.write("Selected model: " + model)
    st.sidebar.write("Set temperature for GPT: " + str(tmp))
    st.sidebar.divider()

    tech_fw = st.sidebar.selectbox("Tech/Framework", ['Django','React','Python','JavaScript'])

    if tech_fw == 'Django':
        topic = st.sidebar.selectbox("Topic", ['middleware','template','using views','models', 'routing', 'restframework'])  # if django selected
    elif tech_fw == 'React':
        topic = st.sidebar.selectbox("Topic", ['hooks','context','redux','routing','components'])
    elif tech_fw == 'Python':
        topic = st.sidebar.selectbox("Topic", ['decorators','generators','classes','functions','modules'])
    else:
        topic = st.sidebar.selectbox("Topic", ['variables','functions','arrays','objects','loops'])

    level = st.sidebar.selectbox("Level", ['senior','mid','junior'])

    ## Model - LLM

    model = "gpt-4o"
    temperature = tmp

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    msgs = StreamlitChatMessageHistory(key="special_app_key")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("In which topic do you want to generate interview task? Select the technology, topic and the level from the sidebar! You can also select the model and temperature for the GPT model.")

    ## Prompt template - role
    system_prompt = f""" 
        You are a {tech_fw} developer. Your job is to provide interview questions related to {topic}.
        Provide the questions in a clear and concise manner. Use bullet points to list the parts of the question. Your answer should be in {lang} language."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{messages}"),
        ]
    )

    chain = prompt | llm

    ## Message history

    store = {}

    with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="messages",
        history_messages_key="history",
    )

    ## Streamlit interface

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    prompt = f"Genarate an interview programming task related to {tech_fw}/{topic} in {level}-level!"

    if "disabled" not in st.session_state:
        st.session_state.disabled = False

    # Disable the submit button after it is clicked
    def disable():
        st.session_state.disabled = True

    if get_question := st.button("Generate Interview", use_container_width=True, on_click=disable, disabled=st.session_state.disabled):
        st.chat_message("human").write(prompt)
        st.divider()
        config = {"configurable": {"session_id": "task"}}
        response = with_message_history.invoke(
            {"messages": prompt},
            config=config,
        )
        st.chat_message("ai").write(response.content)

    if st.session_state.disabled:
        if get_solution := st.button("Solve the task", use_container_width=True):
            solution = "Solve the task!"
            st.chat_message("human").write(solution)
            st.divider()
            config = {"configurable": {"session_id": "solution"}}
            response2 = with_message_history.invoke(
                {"messages": solution},
                config=config,
            )
            st.chat_message("ai").write(response2.content)

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

