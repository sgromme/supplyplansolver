FROM python:3.9-alpine3.20
# python:3.9-alpine3.20
ADD consumer.py .
ENV buildtag=1
ENV TEST=4
# add requirements file
#RUN pip install python-dotenv
# add requirements file
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python", "./consumer.py"]