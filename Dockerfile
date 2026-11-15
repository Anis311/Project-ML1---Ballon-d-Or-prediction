# Web service predicting the probability of a Ballon d'Or top-3 finish.
# The Python version must match "python_version" in the Pipfile.
FROM python:3.14-slim

RUN pip install --no-cache-dir pipenv

WORKDIR /app

# 1. Dependencies first: this layer is rebuilt only when the Pipfile changes
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

# 2. Code and trained model
COPY features.py predict.py model.bin ./

EXPOSE 9696

# gunicorn is a production web server (Flask's own server is for development only)
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
