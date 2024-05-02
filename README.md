# streamlit-openai-chatbot
Quick chatbot to converse with an openai like server api. e.g. works for vllm and llamacpp as well openai's web endpoints.

Just input the correct URL for the endpoint and the correct token in the app.py file. e.g. if running locally then input ENDPOINT_URL = http://0.0.0.0:8000/v1/

The make file shows how to run a server using llamacpp, either using docker or this environment. 
It assumes that the model (Phi-3-mini-4k-instruct-q4.gguf) has been downloaded and saved in a folder called models in this directory. If you would like to use a different model, simply download it and replace the name in the commands.

!['Example interaction](demo.png)