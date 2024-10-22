#!/bin/sh

echo "Creating the topics"
kafka-topics.sh --create --topic quickstart-events --bootstrap-server kafka:9092
kafka-topics.sh --create --topic purchases --bootstrap-server kafka:9092
kafka-topics.sh --list  --bootstrap-server kafka:9092
