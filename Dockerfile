# Dockerfile
# Divisão em etapas visando gerar uma imagem mais leve

# Construção
FROM python:3.11-slim AS builder

# Instala as principais dependências
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Instalando requirements do projeto
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiando dados do projeto
COPY app/ ./app/
COPY data/ ./data/

# Execução
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copiando apenas elementos necessários
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/app ./app
COPY --from=builder /app/data ./data

# Garantir existência da pasta de modelos
RUN mkdir -p ./app/model

# Expondo porta
EXPOSE 8000

# Comando inicial
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
