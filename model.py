from math import ceil
import csv 
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
import sklearn
#from keras.utils.vis_utils import plot_model
lines = []
correction = 0.4

with open('/opt/sim_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

images = []
stearing = []

def get_satu_image(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)[:,:,1]
    alpha=2
    beta=90
    proc_image=cv2.addWeighted(image,alpha,np.zeros(image.shape, image.dtype),0,beta)
    return proc_image.reshape(160,320,1)
    
for line in train_samples:
    ccam_path = line[0]
    file_name = ccam_path.split('/')[-1]
    cent_path = '/opt/sim_data/IMG/'+file_name
    image = get_satu_image(cent_path)    
    images.append(image)
    measurement = float(line[3])
    stearing.append(measurement)
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    stearing.append(measurement_flipped)

    lcam_path = line[1]
    file_name = lcam_path.split('/')[-1]
    left_path = '/opt/sim_data/IMG/'+file_name
    image = get_satu_image(left_path)
    images.append(image)
    measurement = float(line[3])+correction
    stearing.append(measurement)
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    stearing.append(measurement_flipped)

    rcam_path = line[2]
    file_name = rcam_path.split('/')[-1]
    right_path = '/opt/sim_data/IMG/'+file_name
    image = get_satu_image(right_path)
    images.append(image)
    measurement = float(line[3])-correction
    stearing.append(measurement)
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    stearing.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(stearing)

images = []
stearing = []
for line in validation_samples:
    ccam_path = line[0]
    file_name = ccam_path.split('/')[-1]
    cent_path = '/opt/sim_data/IMG/'+file_name
    image = get_satu_image(cent_path)
    images.append(image)
    measurement = float(line[3])
    stearing.append(measurement)   

X_valid = np.array(images)
y_valid = np.array(stearing)


def generator(X_data,y_data,batch_size=128):
    num_samples = len(X_data)
    while 1:# Loop forever so the generator never terminates
        sklearn.utils.shuffle(X_data,y_data)
        for offset in range(0, num_samples, batch_size):
            X_ret = X_data[offset:offset+batch_size]
            y_ret = y_data[offset:offset+batch_size]
            yield sklearn.utils.shuffle(X_ret, y_ret)
            

batch_size = 128
train_generator = generator(X_train,y_train, batch_size=batch_size)
validation_generator = generator(X_valid,y_valid, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,1)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Conv2D(8, (5, 5), padding="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5, 5), padding="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(rate = 50))
model.add(Conv2D(32, (5, 5), padding="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(rate = 50))
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(rate = 50))
model.add(Dense(1))
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#print(model.summary())

model.compile(loss = 'mse',optimizer= 'adam')
history = model.fit_generator(train_generator, steps_per_epoch= ceil(len(X_train)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(X_valid)/batch_size), epochs=5, verbose=1) 

model.save('model.h5')
print(history.history)



