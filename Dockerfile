FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD streamlit run scripts/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
