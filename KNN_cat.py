"""
Finding minimum L1 norm to classify a cat using KNN:

Train set was 5 images: a cat, person, fire hydrant, book, and computer. Test
image was another picture of a cat

Errors in ascending order:

1. 13169896.0 error, Predicted Image of book_1.jpg
2. 14851184.0 error, Predicted Image of cat1_1.jpg
3. 18295371.0 error, Predicted Image of computer_1.jpg
4. 20541269.0 error, Predicted Image of hydrant_1.jpg
5. 21477237.0 error, Predicted Image of person_1.jpg

"""

import numpy as np
import os
import re
import PIL
import shutil
from PIL import Image
from scipy.misc import imread
import scipy

path = "C:/Users/PeterKokalov/lpthw/StanfordCNN/Lecture2/KNN/images/"
directory = os.fsencode(path)

basewidth = 224; baseheight = 224
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith("jpeg"):
        img_path = "{path_dir}{file}".format(path_dir=path, file=filename)
        splits = ".jpg|.jpeg"
        obj = re.split(splits, img_path)[0] + "_1.jpg"

        img = Image.open(img_path).resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
        img.save(obj)

train_dir = "C:/Users/PeterKokalov/lpthw/StanfordCNN/Lecture2/KNN/images/train"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

for file in os.listdir(path):
    filename = os.fsdecode(file)
    if "_1" in filename:
        shutil.move(path + filename, train_dir + "/" + filename)

images = []

def compress(path):
    N_N = scipy.misc.imread(path)
    N_1 = np.ravel(N_N)
    X_i = np.reshape(N_1, (len(N_1), 1))

    return X_i


def create_X_y():
    Xtr = np.array([])
    ytr = np.array([])

    for file in os.listdir(train_dir):
        images.append(file)
        path = train_dir + "/" + file
        arr = compress(path)
        if file != 'test.jpg':
            if len(Xtr) == 0:
                Xtr = arr
            else:
                Xtr = np.hstack((Xtr, arr))
        else:
            ytr = arr

    return Xtr, ytr

X_train, y_train = create_X_y()


class NearestNeighbor(object):
    path = train_dir = "C:/Users/PeterKokalov/lpthw/StanfordCNN/Lecture2/KNN/images/train"

    def __init__(self):
        pass

    
    def predict(self, X, y):

        distances = np.zeros(X.shape[1])

        for ix, col in enumerate(X.T):
            x_i = col.reshape(len(col), 1); err = 0
            for i in range(0, len(x_i)):
                err += np.abs(x_i[i] - y[i])[0]
            distances[ix] = err

        return distances

nn = NearestNeighbor()
L1_norms = nn.predict(X_train, y_train)

print("Errors in ascending order:\n")
for ix, err in enumerate(np.sort(L1_norms)):
    print("{ix}. {err} error, Predicted Image of {img}".format(ix=ix+1, err=err, img=images[ix]))
