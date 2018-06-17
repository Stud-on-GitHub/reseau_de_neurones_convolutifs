# Réseau de Neurones Convolutifs / Convolutional Neural Network



# Part 1 - Construction du CNN


# Import de la librairie Keras et de ces modules
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout



# Initialisation du CNN
classifier = Sequential()


# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, 
                      kernel_size = [3, 3], 
                      strides = 1, 
                      input_shape = (192, 192, 3), 
                      activation = 'relu'))


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Ajout d'une seconde couche de convolution
classifier.add(Conv2D(filters = 32, 
                      kernel_size = [3, 3], 
                      strides = 1,  
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Ajout d'une troisième couche de convolution
classifier.add(Conv2D(filters = 64, 
                      kernel_size = [3, 3], 
                      strides = 1,  
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Ajout d'une quatrième couche de convolution
classifier.add(Conv2D(filters = 64, 
                      kernel_size = [3, 3], 
                      strides = 1,  
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())


# Step 4 - Ajout de la couche complètement connecté

#couche d'entrée
classifier.add(Dense(units = 128, activation = 'relu'))   
classifier.add(Dropout(0.3))

#ajout de couches cachées
classifier.add(Dense(units = 128, activation = 'relu'))   
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 128, activation = 'relu'))   
classifier.add(Dropout(0.3))

#couhce de sortie
classifier.add(Dense(units = 1, activation = 'sigmoid'))   


# Compilation du CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   




# Part 2 - Entrainner le CNN à nos images


from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (192, 192),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (192, 192),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 75,
                         validation_data = test_set,
                         validation_steps = 63)




# Part 3 - Prédiction


# Importation des librairies
import numpy as np
from keras.preprocessing import image


#importation l'image
img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(192, 192))   


#la met en 3 dimensions (rgb)
img = image.img_to_array(img)   


#la met en 4 dimensions (nbr de groupe = 1 img de chien ou de chat testé)
img = np.expand_dims(img, axis=0)   


#fait la prédiction
result = classifier.predict(img)


#précise la class
training_set.class_indices


#teste le résultat
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'






