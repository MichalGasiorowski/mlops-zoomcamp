FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "hw4-MG.py", "./" ]

ENV YEAR_PARAM 2021
ENV MONTH_PARAM 4

RUN [ "sh", "-c", "python hw4-MG.py ${YEAR_PARAM} ${MONTH_PARAM}" ]

