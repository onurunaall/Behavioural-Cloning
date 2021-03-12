import socketio
import eventlet
import base64
import cv2
import numpy as np
from io import BytesIO
from flask import Flask
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

sio = socketio.Server()
app = Flask(__name__) #'__main__'

speed_limit = 10

def img_preprocess(img):
  # cropping unnecessary part of image
  img = img[60:135,:,:] # height,width,channel
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # yuv format due to nvidia arch.
  img = cv2.GaussianBlur(img,(3,3),0) #reducing noise w/GaussianBlur
  img = cv2.resize(img,(200,66)) #for nvidia model arch.
  img = img/255 #normalizing the variables
  return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    #system send the current location of the car
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    #preprocessing the image to use for aoutonomous
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

@sio.on('connect') #message, disconnect #opencom
def connect(sid,environ):
    #as soon as the system is connected, we send initial
    #instruction to the car which is 'stop'
    print('Connected')
    send_control(0,0)

def send_control(steering_angle,throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)),app)
