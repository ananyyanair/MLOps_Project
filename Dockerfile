FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "scripts/streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
