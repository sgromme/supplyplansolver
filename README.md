# supplyplansolver
A prototype of a distributed solver.
The distributed solver will use Kubernetes(or Docker Compose) to deploy and manage the containers.
Apache Kafka will be used for messaging between containers.
Each container will model a node in the supply chain or a node in a graph.


Prerequisite:
Docker Desktop installed
Visual Studio Code installed
python environment setup if you will be developing.


Steps to run:

In the directory docker/consumer run the command "docker build -t consumer ."
In the directory docker/producer run the command "docker build -t producer ."
In the directory supplyplansolver run the command "docker compose up -d"
Observe the container in Docker Desktop
To stop and delete the containers run the command "docker compose down"


