import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import pafy
#from textblob import Textblob
import subprocess

if len(sys.argv) != 2:
    print "Usage: python test.py http://youtube.com/watch?v=XXXXXXXX"

caffe_root = '/home/erik/caffe/'

def getSec(s):
    l = s.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])

# Get youtube URL
url = sys.argv[1]
video = pafy.new(url)

duration = getSec(video.duration)
print "duration: ", duration

likes = video.likes
views = video.viewcount
dislikes = video.dislikes
count = float(likes) + float(dislikes)

approval = float(likes) / count
vote_perc = count / float(views)

# Download video
os.system("youtube-dl -o 'output' " + url)
time.sleep(1)
while(1):
    if os.path.isfile('output.mkv') or os.path.isfile('output.mp4'):
        break
print 'Downloaded video'

# Convert to .avi
if os.path.exists('output.mkv'):
    os.system("ffmpeg -i output.mkv output.avi")
else:
    os.system("ffmpeg -i output.mp4 output.avi")
print 'Converting video'

# Get image every 2 frames
os.system("ffmpeg -i output.avi -r 1 image-%3d.jpeg")
print 'Getting images'

# Resize images to 256x256 pixels
os.system("mogrify -verbose -resize 256x256\! *.jpeg")
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

frame_info = {}
total = 0
for image in os.listdir(os.getcwd()):
    if image.endswith('.jpeg'):
            input_image = caffe.io.load_image(image)
            #plt.imshow(input_image)
            prediction = net.predict([input_image])
            print 'prediction shape: ', prediction[0].shape
            #plt.plot(prediction[0])
            print 'predicted class: ', prediction[0].argmax()

            imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
            labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
            #top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:6:-1]

            frame_label = labels[prediction[0].argmax()]

            image = image[:-5]
            image = image[6:]
            time = (float(image) / (float(duration) + 3)) * float(duration)

            if frame_label not in frame_info:
                frame_info[frame_label] = []
            frame_info[frame_label].append(time)
            #image_labels.append(labels[prediction[0].argmax()] )
            print "time: ", time
            print "IMAGE_LABEL:", labels[prediction[0].argmax()] 
            total += 1

#print image_labels

largest = 1
label_largest = ''
secLargest = 0
label_secLargest = ''
for label in frame_info:
    if len(frame_info[label]) > secLargest:
        secLargest = len(frame_info[label])
        label_secLargest = label
        if len(frame_info[label]) > largest:
            label_secLargest = label_largest
            secLargest = largest
            largest = len(frame_info[label])
            label_largest = label


category = video.category
description = video.description
keywords = video.keywords
mix = video.mix
print 'Most common in video: ', label_largest
print 'Percent of frames that were most popular: ', float(largest) / float(total) * 100
print '2nd most common in video: ', label_secLargest
print 'Percent of people who liked the video, given that they voted: ', approval * 100
print 'Percent of people who voted, given they watched the video: ', vote_perc * 100
print 'Video Category: ', category
print 'Video description: ', description
print 'Video keywords: ', keywords
print "Mix: ", mix



frame_info[label_largest].sort()

val = 0.

axes = plt.gca()
axes.set_xlim(0, duration)

fig = plt.figure()
fig.suptitle(label_largest, fontsize=20)
plt.plot(frame_info[label_largest], np.zeros_like(frame_info[label_largest]) + val, 'x')

plt.show()

os.system("rm -f image-0*")
os.system("rm  output*")
