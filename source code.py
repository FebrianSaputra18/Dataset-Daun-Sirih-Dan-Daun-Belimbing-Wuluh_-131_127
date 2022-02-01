from google.colab import drive
drive.mount('/content/gdrive')

import os

%cd /content/gdrive/MyDrive

# check quantity of images in training and testing using lambda
import os

train_dir = "/content/gdrive/MyDrive/AI"
val_dir = "/content/gdrive/MyDrive/AI"

sirih_path = val_dir + '/DAUN_SIRIH'
belimbing_path = val_dir + '/DAUN_BELIMBING_WULUH'

sirih_len = len(os.listdir(sirih_path))
belimbing_len = len(os.listdir(belimbing_path))

print("jumlah dataset Training : ", sirih_len + belimbing_len)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D

# Feature Extraction Layer
model = Sequential()

model.add(InputLayer(input_shape=[150,150,3]))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=2, padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dropout(0.0001))

# Fully Connected Layer
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.0001))
model.add(Dense(1, activation='sigmoid'))

# Print model summary
print(model.summary())

# Compile model
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.0001), 
              loss='binary_crossentropy', 
              metrics=['acc'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

# setting Callback so we can save the best model in format h5 and i save it in my drive
callbacks = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')        
directory_to_save_best_model_file = train_dir + '/DONE'
best_model = ModelCheckpoint(directory_to_save_best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

history = model.fit(
      train_generator,
      steps_per_epoch=5,  # images = batch_size * steps
      epochs=5,
      validation_data=validation_generator,
      validation_steps=1,  #  images = batch_size * steps
      callbacks = [callbacks, best_model])

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn import metrics

import seaborn as sns
sns.set(style='whitegrid')

# loading model to evaluate more depth
from keras.models import load_model
model_path = train_dir + '/DONE'
model = load_model(model_path)

  def my_metrics(y_true, y_pred):
      accuracy=accuracy_score(y_true, y_pred)
      precision=precision_score(y_true, y_pred,average='weighted')
      f1Score=f1_score(y_true, y_pred, average='weighted') 
      print("Accuracy  : {}".format(accuracy))
      print("Precision : {}".format(precision))
      print("f1Score : {}".format(f1Score))
      cm=confusion_matrix(y_true, y_pred)
      print(cm)
      return accuracy, precision, f1Score

  height=150; width=150
  batch_size=20

  test_datagen = ImageDataGenerator(rescale=1./255)

  TESTING_DIR = train_dir +'/test'

  test_generator = test_datagen.flow_from_directory(TESTING_DIR,
                                                    batch_size=batch_size,                                                             
                                                    target_size=(height, width),
                                                    class_mode= None,
                                                    shuffle=False
                                                    )

  predictions = model.predict_generator(generator=test_generator)
  yPredictions = predictions > 0.5
  true_classes = test_generator.classes
  class_names = test_generator.class_indices
  Cmatrix_test = confusion_matrix(true_classes, yPredictions)

  testAcc,testPrec, testFScore = my_metrics(true_classes, yPredictions)

  plt.figure(figsize=(20,20))
  ax= plt.subplot()
  data = np.asarray(Cmatrix_test).reshape(2,2)
  sns.heatmap(data,annot=True, fmt='',ax=ax, cmap=plt.cm.Reds)
  ax.set_xlabel('Predicted labels')
  ax.set_ylabel('True labels') 
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels(class_names)   
  ax.yaxis.set_ticklabels(class_names)
  plt.title('Confusion Matrix Test',fontsize=14)
  plt.show()

from sklearn.metrics import classification_report
print(classification_report(true_classes, yPredictions, target_names=class_names))

from google.colab import drive
drive.mount('/content/gdrive')

# loading model to evaluate more depth
import numpy as np
from keras.models import load_model
model_path = '/DONE'
model = load_model(model_path)

from google.colab import files
from keras.preprocessing import image

uploaded=files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path= fn
  img=image.load_img(path, target_size=(150, 150))
  
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  
  print(classes[0])
  
  if classes[0]>0:
    print(fn + " is a Daun Belimbing Wuluh")
    
  else:
    print(fn + " is a Daun Sirih")
 
