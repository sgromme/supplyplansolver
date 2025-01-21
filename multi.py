import kafka
from kafka import KafkaConsumer, KafkaProducer
import threading

def consume_messages():
    consumer = KafkaConsumer('purchases', bootstrap_servers=['localhost:9092'])
    for message in consumer:
        print(message.value.decode('utf-8'))

def produce_messages():
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    for _ in range(10):
        producer.send('purchases', b'Hello, Kafka!')
        producer.flush()

if __name__ == '__main__':
    
    print("the beginning of the program")
    consumer_thread = threading.Thread(target=consume_messages)
    producer_thread = threading.Thread(target=produce_messages)
    
    consumer_thread.start()
    producer_thread.start()
    
    consumer_thread.join()
    producer_thread.join()