#!/usr/bin/env python

import os
import threading

from dotenv import load_dotenv
from kafka import KafkaProducer


def produce_messages():
    try:
        producer = KafkaProducer(bootstrap_servers=["kafka:9092"])
        for x in range(100000):
            producer.send("purchases", b"Hello, Kafka!")
            print(f"sent message {x}")

    except Exception as e:
        print(f"Error in producer: {e}")

    producer.flush()


if __name__ == "__main__":
    print("the beginning of the program")
    # get environment variables (from compose.yaml) , not currently using
    print("getting environment variable")
    load_dotenv()
    print("Test=")
    print(os.getenv("TEST"))
    print("buildtag=1")
    print(os.getenv("buildtag"))

    producer_thread = threading.Thread(target=produce_messages)

    producer_thread.start()

    producer_thread.join()
