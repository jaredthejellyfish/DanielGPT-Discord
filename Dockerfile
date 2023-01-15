FROM python:3.10-slim

WORKDIR /bot
COPY . /bot/

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "bot.py"]
