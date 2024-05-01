from langchain_core.messages import AIMessage, HumanMessage # schema for human and ai messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
import requests
import streamlit as st
from streamlit_chat import message
import openai
import os

ENDPOINT_URL = "https://api.openai.com/v1" # uses openai.com by default, use "http://0.0.0.0:8000/v1" for local
API_KEY = os.environ.get('OPENAI_API_KEY')


def list_models():
    r = requests.get('https://api.openai.com/v1/models', headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY")})
    if r.status_code == 200:
        return [d['id'] for d in r.json()['data']]
    else:
        return ['failed to connect']

def get_llm(endpoint_url = None, model = None, temperature = 0.7, max_tokens=500):
    """
    retrieve an llm client from the endpoint
    """
    llm = ChatOpenAI(
        openai_api_base=endpoint_url, 
        model = model,
        temperature=temperature, 
        max_tokens = max_tokens)
    return llm

def get_conversational_chain(endpoint_url = None, model = None, temperature = 0.7, max_tokens=500):
    """
    returns a converstational langchain
    """
    llm = get_llm(endpoint_url, model, temperature, max_tokens)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Please answer the user's questions, taking chat history into account."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return prompt | llm

def initialize_chat_history():
    st.session_state.chat_history = [AIMessage(content="Hello, how can I help you?")]

############################################### APP ###############################################

# IMPORTANT: streamlit runs the whole script everytime an input is registered. 
# checking if an object is in session_state prevents it from doing this

#page config
st.set_page_config(page_title="Chat With Websites", page_icon="🤖")
st.title("LLM Chat")

# save chat history to session state
if "chat_history" not in st.session_state:
    initialize_chat_history()

#sidebar
with st.sidebar: # anything in here will appear in the side bar
    endpoint_url = st.text_input('Endpoint URL', value = ENDPOINT_URL)
    if endpoint_url:
        model_name = st.selectbox('Model Name', list_models())
        if st.button('New Chat?'):
            initialize_chat_history()
    st.header("Settings")
    temperature = st.slider('Temperature', 0.0, 1.0, 0.7)
    max_tokens = st.slider('Max Tokens', 1, 5096, 512)

# Chat App
if model_name and model_name != 'failed to connect':
    # create chain
    chat_chain = get_conversational_chain(
        endpoint_url=endpoint_url, model=model_name, max_tokens=max_tokens, temperature=temperature)
    user_query = st.chat_input("Enter your message here:")
    if user_query is not None and user_query != "":
        # get response
        response = chat_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input" : user_query,
        })
        # add to history
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(response.content))

    # writes conversation to app
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):        
                st.write(message.content)