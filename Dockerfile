# PaperRank on Hugging Face Spaces (Docker SDK, free CPU tier).
FROM python:3.13-slim

# Spaces runs the container as UID 1000, so build and run as that user too;
# otherwise the model cache and pip installs would be owned by root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR $HOME/app

# CPU-only torch first (~200 MB instead of ~2 GB of unused CUDA libraries);
# requirements.txt then sees torch==2.12.1 as already installed.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Bake the default reranker into the image so the first search after a cold
# start doesn't wait on a download. The other two models download on first use.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
