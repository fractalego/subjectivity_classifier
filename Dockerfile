FROM python:3.6-buster
# Download git and the repo
RUN apt-get update
RUN apt-get install -y git nano
RUN git clone https://github.com/fractalego/subjectivity_classifier.git
# Download requirements
RUN cd subjectivity_classifier && pip install -r requirements.txt
# Download Dataset
RUN cd subjectivity_classifier/data/word_embeddings/ && wget -nc -O glove.6B.50d.txt http://nlulite.com/download/glove
RUN python -m nltk.downloader 'punkt'
#Set the workdir
WORKDIR /subjectivity_classifier
#Command line usage: docker run <container-id> <any text>
ENTRYPOINT ["python", "-m", "subjectivity.classify"]
