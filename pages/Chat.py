## $ streamlit run <app_name>.py
## deploy -> https://streamlit.io/cloud

## https://api.python.langchain.com/en/latest/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html

import os

import streamlit as st
from environs import Env
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_huggingface import HuggingFaceEndpoint
from auth import auth


# Run authentication first
authenticator = auth()
login = authenticator.login('main', fields = {'Form name': 'Login'})
name, authentication_status, username = login

## Environment variables
env = Env()
env.read_env()

## Langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = env.str("LANGCHAIN_API_KEY")

## Style
# with open('style.css') as f:
#     css = f.read()
# st.write(f'<style>{css}</style>', unsafe_allow_html=True)

## Content
## Check if the user is authenticated
if authentication_status:
    
    ## Sidebar
    st.sidebar.success("OpenAI Chatbot")
    st.sidebar.write(f'Welcome *{username}*', '👋')
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.divider()

    # Model selection
    select_model = st.sidebar.selectbox("Select model", 
        [
        'gpt-4o',
        'gpt-4',
        'gpt-3.5-turbo',
        'claude-3-5-sonnet-20240620',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        ]
    )

    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tmp = st.sidebar.select_slider(
        "Select temperature",
        options=scale,
        value=0.3
    )
    st.sidebar.divider()
    st.sidebar.write("Selected model: " + select_model)
    st.sidebar.write("Selected temperature for GPT: " + str(tmp))

    ## Model - LLM
    model = select_model
    temperature = tmp

    if 'gpt' in model:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"]
        )
    elif 'claude' in model:
        llm = ChatAnthropic(
            model=model,
            temperature=0,
            #max_tokens=512,
            timeout=None,
            max_retries=2,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            # base_url="...",
            # other params...
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            #max_new_tokens=512,
            do_sample=False,
            huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"],
        )

    msgs = StreamlitChatMessageHistory(key="special_app_key")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you today?")

    ## Prompt template - role
    system_prompt = """ You are a helpful assistant. Answer all questions to the best of your ability in. """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{messages}"),
        ]
    )

    chain = prompt | llm

    ## Message history
    with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs, 
        input_messages_key="messages",
        history_messages_key="history",
    )

    ## Streamlit interface
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        ## As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        config = {"configurable": {"session_id": "any"}}
        response = with_message_history.invoke(
            {"messages": prompt},
            config=config,
        )
        if 'gpt' in model:
            response = response.content
        st.chat_message("ai").write(response)
    
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')