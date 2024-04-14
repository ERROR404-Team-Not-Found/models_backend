FROM python:3.11-alpine

WORKDIR /app

COPY . .

ENV MINIO_ENDPOINT = "minio1.sedimark.work"
ENV MINIO_ACCESS_KEY = "super"
ENV MINIO_SECRET_KEY = "doopersecret"
ENV TRAIN_PATH = "/home/ubuntu-machine/train.py"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]