FROM continuumio/anaconda3:latest

RUN pip install --upgrade pip

# Install production dependencies.
ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

EXPOSE 5000
CMD [ "python", "./app.py" ]