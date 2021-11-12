FROM python:3.8.7-slim-buster
COPY ./docker-flask-deployment /usr/local.python
WORKDIR /usr/local.python
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    unixodbc \
    libpq-dev 
RUN apt-get install -y g++ unixodbc-dev
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD python iris_pred_api.py