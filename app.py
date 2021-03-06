import random
from flask import Flask, request
from pymessenger.bot import Bot
import requests
from io import BytesIO
import flask
import sys
import os
import glob
import re
import numpy as np
#from pathlib import Path
import wikipedia as wk
import json
import pickle


# Import fast.ai Library
from fastai import *
from fastai.vision.all import *

 # Initializing our Flask application
app = Flask(__name__)      

ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
VERIFY_TOKEN = os.environ['VERIFY_TOKEN']
bot = Bot(ACCESS_TOKEN)

# Importing standard route and two requst types: GET and POST.
# We will receive messages that Facebook sends our bot at this endpoint
@app.route('/', methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)

    else:
            # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    # Facebook Messenger ID for user so we know where to send response back to
                    recipient_id = message['sender']['id']
                    if message['message'].get('text'):
                        response_sent_text = get_message()
                        send_message(recipient_id, response_sent_text)
                    # if user send us a GIF, photo, video or any other non-text item
                    if message['message'].get('attachments'):
                        if message['message']['attachments'][0]['type'] == "image":
                            image_url = message["message"]["attachments"][0]["payload"]["url"]
                            pred_message = model_predict(image_url)
                            send_message(recipient_id, pred_message)
                    
    return "Message Processed"


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'

#return random response if user sends text
def get_message():
    sample_responses = ["Sorry I'm not smart enough to engage in natural conversation yet, I only understand images of fruit", 
    "Please upload an image of a fruit",
                        "I cannot understand words just yet, please upload an image of a fruit"] 
                        
    return random.choice(sample_responses)


# Uses PyMessenger to send response to the user
def send_message(recipient_id, response):
    bot.send_text_message(recipient_id, response)
    return "success"

path = Path()
Path().ls(file_exts='.pkl')
learn = load_learner(path/'models/export34.pkl')


# Process the image and prediction
@app.route('/analyse', methods=['GET', 'POST'])
def model_predict(url):
    response = requests.get(url)
    img = PILImage.create(BytesIO(response.content))
    prediction = learn.predict(img)[0]
    img_message = str(prediction)
    pred_result= (f'prediction: {img_message}\n')
            
    return pred_result


if __name__ == "__main__":
    app.run()


# comment
''' Below is the predict function with wikipedia included,I had to remove it because
some predicted classes were throwing a missing id exception and crashing the app,
error handling needs to be done before it can be properly used '''


'''
def model_predict(url):
    response = requests.get(url)
    img = PILImage.create(BytesIO(response.content))
    prediction = learn.predict(img)[0]
    img_message = str(prediction)
    wiki_msg = re.sub("\d+\s\d+\.\d+", "", img_message)
    wiki_info = wk.summary(wiki_msg, sentences = 3)
    wiki_result=(f'Result: {img_message}\n'
            f'\n'
           f'Summary: {wiki_info}')
    return wiki_result
'''