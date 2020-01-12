import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers.merge import Concatenate
from keras import backend as K

from os import listdir
import numpy as np
import cv2
import sys
import time
from PIL import Image
import matplotlib.pyplot as plt
#-----------------------
#TRY OF SOLUTION because of bug no.
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
#
dump = False

color = (67,67,67)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(96, 96))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)



    img = preprocess_input(img)
    return img

#------------------------

def builtModel():
	myInput = Input(shape=(96, 96, 3))

	x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = MaxPooling2D(pool_size=3, strides=2)(x)
	x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_1')(x)
	x = Conv2D(64, (1, 1), name='conv2')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
	x = Activation('relu')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2D(192, (3, 3), name='conv3')(x)
	x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
	x = Activation('relu')(x)
	Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_2')(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = MaxPooling2D(pool_size=3, strides=2)(x)

	# Inception3a
	inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
	inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
	inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
	inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
	inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
	inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
	inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

	inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
	inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
	inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
	inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
	inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
	inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
	inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

	inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
	inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
	inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
	inception_3a_pool = Activation('relu')(inception_3a_pool)
	inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

	inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
	inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
	inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

	inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

	# Inception3b
	inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
	inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
	inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
	inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
	inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
	inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
	inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

	inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
	inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
	inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
	inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
	inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
	inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
	inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

	inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
	inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
	inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
	inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
	inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
	inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
	inception_3b_pool = Activation('relu')(inception_3b_pool)
	inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

	inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
	inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
	inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

	inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

	# Inception3c
	inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name='inception_3c_3x3_conv1')(inception_3b)
	inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn1')(inception_3c_3x3)
	inception_3c_3x3 = Activation('relu')(inception_3c_3x3)
	inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
	inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name='inception_3c_3x3_conv'+'2')(inception_3c_3x3)
	inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn'+'2')(inception_3c_3x3)
	inception_3c_3x3 = Activation('relu')(inception_3c_3x3)

	inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name='inception_3c_5x5_conv1')(inception_3b)
	inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn1')(inception_3c_5x5)
	inception_3c_5x5 = Activation('relu')(inception_3c_5x5)
	inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
	inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name='inception_3c_5x5_conv'+'2')(inception_3c_5x5)
	inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn'+'2')(inception_3c_5x5)
	inception_3c_5x5 = Activation('relu')(inception_3c_5x5)

	inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
	inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

	inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

	#inception 4a
	inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_4a_3x3_conv'+'1')(inception_3c)
	inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'1')(inception_4a_3x3)
	inception_4a_3x3 = Activation('relu')(inception_4a_3x3)
	inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
	inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name='inception_4a_3x3_conv'+'2')(inception_4a_3x3)
	inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'2')(inception_4a_3x3)
	inception_4a_3x3 = Activation('relu')(inception_4a_3x3)

	inception_4a_5x5 = Conv2D(32, (1,1), strides=(1,1), name='inception_4a_5x5_conv1')(inception_3c)
	inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn1')(inception_4a_5x5)
	inception_4a_5x5 = Activation('relu')(inception_4a_5x5)
	inception_4a_5x5 = ZeroPadding2D(padding=(2,2))(inception_4a_5x5)
	inception_4a_5x5 = Conv2D(64, (5,5), strides=(1,1), name='inception_4a_5x5_conv'+'2')(inception_4a_5x5)
	inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn'+'2')(inception_4a_5x5)
	inception_4a_5x5 = Activation('relu')(inception_4a_5x5)

	inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
	inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
	inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
	inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)

	inception_4a_pool = Conv2D(128, (1,1), strides=(1,1), name='inception_4a_pool_conv'+'')(inception_4a_pool)
	inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_pool_bn'+'')(inception_4a_pool)
	inception_4a_pool = Activation('relu')(inception_4a_pool)
	inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

	inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_4a_1x1_conv'+'')(inception_3c)
	inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_1x1_bn'+'')(inception_4a_1x1)
	inception_4a_1x1 = Activation('relu')(inception_4a_1x1)

	inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

	#inception4e
	inception_4e_3x3 = Conv2D(160, (1,1), strides=(1,1), name='inception_4e_3x3_conv'+'1')(inception_4a)
	inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'1')(inception_4e_3x3)
	inception_4e_3x3 = Activation('relu')(inception_4e_3x3)
	inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
	inception_4e_3x3 = Conv2D(256, (3,3), strides=(2,2), name='inception_4e_3x3_conv'+'2')(inception_4e_3x3)
	inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'2')(inception_4e_3x3)
	inception_4e_3x3 = Activation('relu')(inception_4e_3x3)

	inception_4e_5x5 = Conv2D(64, (1,1), strides=(1,1), name='inception_4e_5x5_conv'+'1')(inception_4a)
	inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'1')(inception_4e_5x5)
	inception_4e_5x5 = Activation('relu')(inception_4e_5x5)
	inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
	inception_4e_5x5 = Conv2D(128, (5,5), strides=(2,2), name='inception_4e_5x5_conv'+'2')(inception_4e_5x5)
	inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'2')(inception_4e_5x5)
	inception_4e_5x5 = Activation('relu')(inception_4e_5x5)

	inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
	inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

	inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

	#inception5a
	inception_5a_3x3 = Conv2D(96, (1,1), strides=(1,1), name='inception_5a_3x3_conv'+'1')(inception_4e)
	inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn'+'1')(inception_5a_3x3)
	inception_5a_3x3 = Activation('relu')(inception_5a_3x3)
	inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
	inception_5a_3x3 = Conv2D(384, (3,3), strides=(1,1), name='inception_5a_3x3_conv'+'2')(inception_5a_3x3)
	inception_5a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_3x3_bn'+'2')(inception_5a_3x3)
	inception_5a_3x3 = Activation('relu')(inception_5a_3x3)

	inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
	inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
	inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
	inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)

	inception_5a_pool = Conv2D(96, (1,1), strides=(1,1), name='inception_5a_pool_conv'+'')(inception_5a_pool)
	inception_5a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_pool_bn'+'')(inception_5a_pool)
	inception_5a_pool = Activation('relu')(inception_5a_pool)
	inception_5a_pool = ZeroPadding2D(padding=(1,1))(inception_5a_pool)

	inception_5a_1x1 = Conv2D(256, (1,1), strides=(1,1), name='inception_5a_1x1_conv'+'')(inception_4e)
	inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5a_1x1_bn'+'')(inception_5a_1x1)
	inception_5a_1x1 = Activation('relu')(inception_5a_1x1)

	inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

	#inception_5b
	inception_5b_3x3 = Conv2D(96, (1,1), strides=(1,1), name='inception_5b_3x3_conv'+'1')(inception_5a)
	inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn'+'1')(inception_5b_3x3)
	inception_5b_3x3 = Activation('relu')(inception_5b_3x3)
	inception_5b_3x3 = ZeroPadding2D(padding=(1,1))(inception_5b_3x3)
	inception_5b_3x3 = Conv2D(384, (3,3), strides=(1,1), name='inception_5b_3x3_conv'+'2')(inception_5b_3x3)
	inception_5b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_3x3_bn'+'2')(inception_5b_3x3)
	inception_5b_3x3 = Activation('relu')(inception_5b_3x3)

	inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)

	inception_5b_pool = Conv2D(96, (1,1), strides=(1,1), name='inception_5b_pool_conv'+'')(inception_5b_pool)
	inception_5b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_pool_bn'+'')(inception_5b_pool)
	inception_5b_pool = Activation('relu')(inception_5b_pool)

	inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

	inception_5b_1x1 = Conv2D(256, (1,1), strides=(1,1), name='inception_5b_1x1_conv'+'')(inception_5a)
	inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_5b_1x1_bn'+'')(inception_5b_1x1)
	inception_5b_1x1 = Activation('relu')(inception_5b_1x1)

	inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

	av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
	reshape_layer = Flatten()(av_pool)
	dense_layer = Dense(128, name='dense_layer')(reshape_layer)
	norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

	# Final Model
	model = Model(inputs=[myInput], outputs=norm_layer)
	return model

model = builtModel()
print("El modelo ha sido exitosamente cargado")

#------------------------


model.load_weights('weights/weights.h5')
print("weights loaded")

from scipy import spatial
#------------------------
def findCosineDistance(source_representation, test_representation):
    #a = np.matmul(np.transpose(source_representation), test_representation)
    #b = np.sum(np.multiply(source_representation, source_representation))
    #c = np.sum(np.multiply(test_representation, test_representation))
    #return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
	return spatial.distance.cosine(source_representation, test_representation)
#To replace function above with scipy.spatial.distance.cosine(soure_representation, test_representation)


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = np.subtract(source_representation,test_representation)
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    #euclidean_distance = l2_normalize(euclidean_distance)
    return euclidean_distance

# -*******************************************************************************************************************-

#-*********************************************************************************************************



#------------------------

metric = "cosine"  #cosine, euclidean

if metric == "cosine":
	threshold = 0.45
else:
	threshold = 0.95


#------------------------
def faceRecognition():
	model = builtModel()
	print("The model has been built")
	model.load_weights('weights/weights.h5')
	print("weights loaded")


	img = preprocess_image('user_face.jpg')
	representation = model.predict(img)

	print('Username photoId retrieved successfully')
	#------------------------

	cap = cv2.VideoCapture(0) #webcam
	print('here we go again...')
	timer =  5
	permiso = False

	while(True):

		ret, img = cap.read()
		faces = face_cascade.detectMultiScale(img, 1.3, 5)

		if permiso == True:
			retJson = {
						'status' : 200,
						'msg': 'Welcome again!'
					}
			break


		for (x,y,w,h) in faces:
			if timer == -1 :
				permiso = True
				break

			elif w > 100:

				cv2.rectangle(img, (x,y), (x+w, y+h) , color , 1)

				detected_face = img[int(y): int(y+h), int(x):int(x+w)]
				detected_face = cv2.resize(detected_face, (96,96))

				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 127.5
				img_pixels -= 1

				user_captured = model.predict(img_pixels)[0,:]
				distances = []
				user_source = representation
				#print('First checkpoint')
				if metric == "cosine":
					distance = findCosineDistance(user_captured, user_source)
				elif metric == "euclidean":
					distance = findEuclideanDistance(user_captured, user_source)

				label_name =  'user'  + ' not recognized'

				umbral = 99.55


				if distance <= threshold:
					if metric == "euclidean":
						similarity = 100 + (90 - 100* distance)
					elif metric == "cosine":
						similarity = 100 + (40 - 100* distance)

					if similarity > 99.99: similarity = 99.45



					if similarity >= umbral:
						label_name = str(timer) + ' steps verification remaining '
						timer = timer - 1
						time.sleep(0.5)
						print(umbral ,timer )
						if timer ==3:
							umbral= 99.65
							print(umbral , timer )
						elif timer ==2:
							umbral= 99.75
							print(umbral , timer )
						elif timer ==1:
							umbral = 99.85
							print(umbral , timer )
						elif timer == 0:
							umbral = 99.989
							print(umbral , timer )



					else:
						label_name = str(timer) + ' steps verification remaining'
						#timer = 5

				else:
					timer = 5
					print('Excuse me ' + 'user'  + ', could you come a little bit closer? :)')


				#print('Second checkpoint')
				cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2 )

				cv2.line(img,(x+w, y-64),(x+w-25, y-64),color,1) #1
				cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),color,1) #2


				cv2.imshow('img', img)



				if cv2.waitKey(1) & 0xFF == ord('q'):
					retJson = {
								'status' : 300,
								'msg':  "We're sorry, we couldn't recognize you yet"
					}
					break






	time.sleep(3.0)
	cap.release()
	cv2.destroyAllWindows()

	return retJson
