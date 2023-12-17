#將從訓練集中提取類別對應的名稱
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, csv
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#model
#model structure
model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
#建立後段的分類層，兩個全連接，最後softmax
flatten = tf.keras.layers.Flatten(name='flatten')(model.output)
fc1 = tf.keras.layers.Dense(4096, activation='relu',name='fc1')(flatten)
fc2 = tf.keras.layers.Dense(4096, activation='relu',name='fc2')(fc1)
out = tf.keras.layers.Dense(300, activation='softmax')(fc2)
model = tf.keras.Model(model.input, out)
model.summary()

generator = ImageDataGenerator()
model.load_weights('best.h5')
results = generator.flow_from_directory('./train&test/train/')
dicts=results.class_indices
labels=[None]*300
for key in dicts:
    labels[dicts[key]]=key
print(labels)

#Make predictions and convert values ​​into corresponding categories
test_datagen2 = ImageDataGenerator()
files= os.listdir("./grading_data")
k=1
prediction_n = []
for file in files:
    if not os.path.isdir(file):
        print(file)
        img = image.load_img(path=("./grading_data/"+str(file)))
        img = image.img_to_array(img)
        x = K.expand_dims(img,axis=0)       
        preds = model.predict(x,steps=1)
        predname=K.eval(K.argmax(preds))
        name=labels[predname[0]]
        prediction_n.append([file,name])
        print("prediction : ",name.strip())
print(prediction_n)

#write the prediction result into the csv file
i = 0 
data_sorted=sorted(prediction_n,key=lambda filename: int(filename[0][:-4]))
#print(data_sorted)
with open('grading_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image', 'specie'])
    while i < len(prediction_n):
        writer.writerow([data_sorted[i][0],data_sorted[i][1]])
        i=i+1
    print(i)