# Not tested, just a try
FROM python:3.9

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
WORKDIR . .

ENTRYPOINT ["python"]
CMD ["src\main\python.com.eXXcellent.predictive_analytics\predictive_analytics.py"]


