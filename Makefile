install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

launch:
	streamlit run app.py --server.port=8501 --server.address=0.0.0.0

launch-llama-cpp:
	python3 -m llama_cpp.server --model models/Phi-3-mini-4k-instruct-q4.gguf --host 0.0.0.0 --port 8000

launch-llamacpp-docker:
	docker run --rm -it --cap-add SYS_RESOURCE -e USE_MLOCK=0 -p 8000:8000 \
		 -v $(pwd)/models:/models -e Phi-3-mini-4k-instruct-q4.gguf ghcr.io/abetlen/llama-cpp-python:latest

