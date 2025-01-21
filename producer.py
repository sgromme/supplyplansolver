from time import sleep
from json import dumps
from kafka import KafkaProducer


producer = KafkaProducer(bootstrap_servers=['localhost:9094'],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))

print("the beginning of the program")


if __name__ == '__main__':

    print("running main")

    for e in range(1000):
        data = {'number' : e}
        producer.send('purchases', value=data)
        sleep(5)


    print("the end of the program")

