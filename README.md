# supplyplansolver
A prototype of a distributed solver.
The distributed solver will use Kubernetes(or Docker Compose) to deploy and manage the containers.
Apache Kafka will be used for messaging between containers.
Each container will model a node in the supply chain or a node in a graph.


# Prerequisite:
Docker Desktop installed
Visual Studio Code installed
python environment setup if you will be developing.


# Steps to run:

* In the directory source/consumer run the command "docker build -t consumer ."
* In the directory source/producer run the command "docker build -t producer ."
* In the directory supplyplansolver run the command "docker compose up -d"
* Observe the container in Docker Desktop
* To stop and delete the containers run the command "docker compose down"


# Instructions to run from Docker Hub Repository:

* build with a tag or version and use your Docker Hub Username (yourdockerhubusername)
docker build -t yourdockerhubusername/consumer:1.1.1 .
docker build -t yourdockerhubusername/producer:1.1.1 .

* Push images to your Docker Hub
docker push yourdockerhubusername/consumer:1.1.1
docker push yourdockerhubusername/producer:1.1.1

* update compose.yaml to reference the new docker name and tag
consumer:
    image: 'yourdockerhubusername/consumer:1.1.1'
 producer:
    image: 'yourdockerhubusername/producer:1.1.1'

* run the docker compose






