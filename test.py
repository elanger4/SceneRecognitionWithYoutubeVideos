import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import subprocess

if len(sys.argv) != 2:
    print "Usage: python test.py http://youtube.com/watch?v=XXXXXXXX"

caffe_root = '/home/erik/caffe/'

# Get youtube URL
url = sys.argv[1]

# Download video
os.system("youtube-dl -o 'output' " + url)
time.sleep(1)
while(1):
    if os.path.isfile('output.mkv'):
        break
print 'Downloaded video'

# Convert to .avi
os.system("ffmpeg -i output.mkv output.avi")
print 'Converting video'

# Get image every 2 frames
os.system("ffmpeg -i output.avi -r .5 image-%3d.jpeg")
print 'Getting images'

# Resize images to 256x256 pixels
os.system("sudo mogrify -verbose -resize 256x256\! *.jpeg")
print 'Resizing images'

sys.path.insert(0, caffe_root + 'python')

import caffe
os.chdir('../bigred/')

print 'learning'

MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
        mean=np.load(MEAN).mean(1).mean(1),
        channel_swap=(2,1,0),
        raw_scale=255,
        image_dims=(256, 256))

for image in os.listdir(os.getcwd()):
    if image.endswith('.jpeg'):
            input_image = caffe.io.load_image(image)
            plt.imshow(input_image)
            prediction = net.predict([input_image])
            print 'prediction shape: ', prediction[0].shape
            plt.plot(prediction[0])
            print 'predicted class: ', prediction[0].argmax()

            imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
            labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

            top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:6:-1]


            print labels[prediction[0].argmax()] 
