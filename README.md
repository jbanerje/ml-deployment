# ml-deployment

Machine Learning Deployment

1. docker-flask-deployment : Deployment of a basic flask application for iris
    * How to run the Docker Container
        * Make sure the DOCKERFILE is outside the application folder (else have to change the copy file structures in dockerfile)
        * Go the the directory of Docker File and run --> docker build --no-cache -t docker-iris-dep .
        * Once the images is sucessfully created run--> docker run -d -p 5000:5000 docker-iris-dep
        * If running from local computer then connection should be made to localhost:5000
        * Postman Configuration:
            -- POST : http://localhost:5000/iris-pred-api
            -- BODY
                {
                    "sepal_length": 5.7,
                    "sepal_width": 2.5,
                    "petal_length": 5.1,
                    "petal_width": 2.1
                }
            -- HEADERS : Content-type : Application JSON


2. docker-streamlit-deployment : Deployment of a streamlit application for iris
    * How to run the Docker Container
        * Make sure the DOCKERFILE is outside the application folder (else have to change the copy file structures in dockerfile)
        * Go the the directory of Docker File and run --> docker build --no-cache -t docker-iris-stlit-dep .
        * Once the images is sucessfully created run--> docker run -d -p 8501:8501 docker-iris-stlit-dep
        * On the browser use <computer_ip>:8501 to launch the application
