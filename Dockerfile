FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN mkdir -p output

CMD ["sh", "-c", "if [ -f scripts/setup.sh ]; then chmod +x scripts/setup.sh && scripts/setup.sh; fi; if [ -f inference.py ]; then python inference.py; elif [ -f scripts/inference.py ]; then python scripts/inference.py; elif [ -f test.py ]; then python test.py; elif [ -f scripts/test.py ]; then python scripts/test.py; else echo \"No inference entrypoint found\" && exit 2; fi"]

