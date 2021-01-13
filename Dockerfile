FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip 
    
WORKDIR /app
COPY  . /app
RUN pip3 install -r requirements.txt
RUN apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["app.py"]
