import kafka
from kafka import KafkaConsumer, KafkaProducer
import threading

from dotenv import load_dotenv
import os


def consume_messages():
    try:
        consumer = KafkaConsumer(
           'purchases',
            bootstrap_servers=['kafka:9092'],
            auto_offset_reset='earliest',
            group_id='my-group')
       
        for message in consumer:
            print(message.value.decode('utf-8'))
    except Exception as e:
        print(f"Error in consumer: {e}")

    

if __name__ == '__main__':
    
    print("the beginning of the program")

    # get environment variables (from compose.yaml) , not currently using
    print("getting environment variable")
    load_dotenv()
    print("Test=")
    print(os.getenv("TEST"))
    print("buildtag=1")
    print(os.getenv("buildtag"))

    consumer = KafkaConsumer(
           'purchases',
            bootstrap_servers=['kafka:9092'],
            auto_offset_reset='earliest',
            group_id='my-group')
    
    #TODO: need to decode the message
    #TODO: Add multiple process and add to FASTAPI
    try:
        while True:
            msg = consumer.poll(timeout_ms=1.0)
            if msg is None:
                print("no new messages")
            else:
            # TODO:need to decode the message
                print(msg)
 
    
    except Exception as e:
        print(f"Error in consumer: {e}")

    consumer.close()

    print("the end of the program")

