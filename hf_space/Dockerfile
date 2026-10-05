FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only PyTorch keeps the image small; switch the index URL for GPU hardware.
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user setuptools wheel cython -r requirements.txt

# pytorch_medical_imaging builds Cython extensions against the numpy installed above
RUN pip install --no-cache-dir --user --no-build-isolation \
        "git+https://github.com/ml-w/pytorch_medical_imaging@master"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.maxUploadSize=1024", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
