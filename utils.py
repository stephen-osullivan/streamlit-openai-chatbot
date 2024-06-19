from langchain_community.llms import HuggingFaceEndpoint, vllm, ollama
from langchain_openai import ChatOpenAI

import os

def get_model(
        framework : str = 'vllm', 
        model : str = None, 
        endpoint_url : str = None,
        temperature : float = 0.7,
        max_tokens : int = 512,
        ):
    """
    takes in a framework and model and then returns a chain
    """
    if framework == 'openai_compatible':
        llm = get_openai_model(model=model, endpoint_url= endpoint_url, temperature=temperature)
    elif framework == 'huggingface':
        llm = get_hugginface_model(model=model, temperature=temperature)
    elif framework == "vllm":
        llm = get_vllm_model(model=model, endpoint_url=endpoint_url, temperature=temperature)
    elif framework == "ollama":
        llm = get_ollama_model(model=model, temperature=temperature)
    else:
        raise NotImplementedError
    
    return llm 

def get_ollama_model(model = 'llama3',  temperature=0.7):
    llm = ollama.Ollama(
        model=model, 
        stop=['<|eot_id|>'], 
        num_ctx = 1024, 
        temperature=temperature)
    return llm

def get_openai_model(
        model='gpt-3.5-turbo', 
        endpoint_url = 'https://api.openai.com/v1/',
        temperature=0.7):
    if model is None:
        model = 'gpt-3.5-turbo'
    token = os.environ.get('OPENAI_API_KEY', "dummykey")
    if token:
        llm = ChatOpenAI(
            model=model, 
            temperature=temperature,
            max_tokens = 1024,
            openai_api_base=endpoint_url
            )
    else:
        raise Exception('OPENAI_API_KEY NOT PROVIDED')
    return llm

def get_hugginface_model(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature = 0.7):
    
    llm = HuggingFaceEndpoint(
        repo_id=model,  
        temperature=temperature,
        max_new_tokens=1024,
        stop_sequences = ["<|eot_id|>"])

    return llm

def get_vllm_model(model, endpoint_url, temperature = 0.7):
    llm = vllm.VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=endpoint_url,
        model_name=model, 
        temperature=temperature,
        max_tokens = 1024,
        model_kwargs=dict(stop=['<|im_end|>', '<|im_start|>','</s>', "[INST]", "Human:"])
        )
    return llm
