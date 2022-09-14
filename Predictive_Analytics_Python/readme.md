# Predictive Analytics Framework

## How to start the microservice:
    - Option 1: run the application via python
            at src/main/python/com.eXXcellent.predictive_analytics
            run the command 'python app.py'
            the microservice will run on the host address and port defined in the config.py file
    - Option 2: run the application via flask
            at src/main/python/com.eXXcellent.predictive_analytics
            run the command 'flask run --host=<ip-addresse> --port=<port>'
            for example 'flask run --host=0.0.0.0 --port=8080'
            without the definition of the port, the microservice will run on the flask default port 5000