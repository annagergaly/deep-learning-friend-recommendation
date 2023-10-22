FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update
RUN apt install -y

ENV HOME /home/user
	
COPY . .

EXPOSE 8887

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 user

CMD ["bash"]