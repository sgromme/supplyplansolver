from kafka import KafkaConsumer, KafkaProducer

print("the beginning of the program")

producer = KafkaProducer(bootstrap_servers='localhost:9092')

for _ in range(10):
        producer.send('purchases', b'Hello, Kafka!')
        producer.flush()


print("the end of the program")
