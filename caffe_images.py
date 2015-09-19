import numpy as np
import sys
import os
import matplotlib.pyplot as plt

caffe_root = '/home/erik/caffe/'

sys.path.insert(0, caffe_root + 'python')

import caffe
os.chdir('../bigred/')

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
            prediction = net.predict([input_image])
            print 'prediction shape: ', prediction[0].shape
            plt.plot(prediction[0])
            print 'predicted class: ', prediction[0].argmax()

            imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
            labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

            top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:6:-1]
            print labels[prediction[0].argmax()]
#            print 'top_k: ', top_k

