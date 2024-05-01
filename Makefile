install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

launch:
	streamlit run app.py --server.port=8501 --server.address=0.0.0.0