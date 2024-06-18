
from langchain_core.messages import AIMessage, HumanMessage # schema for human and ai messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
import requests
import streamlit as st
from streamlit_chat import message
import openai

from utils import get_model

import os

ENDPOINT_URL = "https://api.openai.com/v1" # uses openai.com by default, use "http://0.0.0.0:8000/v1" for local
API_KEY = os.environ.get('OPENAI_API_KEY', 'dummy_token') # only needed if using openai, not needed for local


def list_models(endpoint_url):
    """
    list the models available at the endpoint
    """
    r = requests.get(os.path.join(endpoint_url, 'models'), headers={"Authorization": f"Bearer {API_KEY}"})
    if r.status_code == 200:
        return [d['id'] for d in r.json()['data']]
    else:
        return ['failed to connect']

def get_conversational_chain(
        framework='vllm',
        endpoint_url = None, 
        model_name = None, 
        temperature = 0.7, 
        max_tokens=500, 
        support_system_message=False):
    """
    returns a converstational langchain
    """

    llm = get_model(framework=framework, endpoint_url=endpoint_url, model=model_name, temperature=temperature)

    if support_system_message:
        system_message = [("system", "Please answer the user's questions, taking chat history into account.")]
    else:
        system_message = [
            HumanMessage(content= "Please answer the user's questions, taking chat history into account."),
            AIMessage(content="OK.")
            ]
    prompt = ChatPromptTemplate.from_messages([
        *system_message,
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()

def initialize_chat_history():
    st.session_state.chat_history = [AIMessage(content="Hello, how can I help you?")]

############################################### APP ###############################################

# IMPORTANT: streamlit runs the whole script everytime an input is registered. 
# checking if an object is in session_state prevents it from doing this

#page config
st.set_page_config(page_title="LLM Chat App", page_icon="🤖")
st.title("LLM Chat")

if st.button('New Chat?'):
    initialize_chat_history()

# save chat history to session state
if "chat_history" not in st.session_state:
    initialize_chat_history()

#sidebar
with st.sidebar: # anything in here will appear in the side bar
    framework = st.selectbox('Framework', ['vllm', 'openai_compatible', 'huggingface'])
    if framework in ['vllm', 'openai_compatible']:
        endpoint_url = st.text_input('Endpoint URL', value = ENDPOINT_URL)
        if endpoint_url:
            model_name = st.selectbox('Model Name', list_models(endpoint_url=endpoint_url))
    elif framework == 'huggingface':
            endpoint_url=None
            model_name = st.selectbox(
                'Model', 
                ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "meta-llama/Meta-Llama-3-8B-Instruct"])
        
    st.header("Settings")
    temperature = st.slider('Temperature', 0.0, 1.0, 0.7)
    max_tokens = st.slider('Max Tokens', 1, 5096, 512)
    is_mistral = st.selectbox('Mistral Model?', [True, False])

# Chat App
if model_name and model_name != 'failed to connect':
    # create chain
    chat_chain = get_conversational_chain(
        framework=framework,
        endpoint_url=endpoint_url, 
        model_name=model_name, 
        max_tokens=max_tokens, 
        temperature=temperature,
        support_system_message= ~is_mistral)
    
    # writes past conversation to app
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):        
                st.write(message.content)

    # get user query
    user_query = st.chat_input("Enter your message here:")
    if user_query is not None and user_query != "":
        with st.chat_message("Human"):
            st.write(user_query)
        # append query to chat history
        st.session_state.chat_history.append(HumanMessage(user_query))

        # get response
        chain_inputs = {
            "chat_history": st.session_state.chat_history,
            "input" : user_query,
        }
        
        # write response stream
        with st.chat_message("AI"):
            response = st.write_stream(chat_chain.stream(chain_inputs))
        # append response to chat history
        st.session_state.chat_history.append(AIMessage(response))