#first we will create the model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=25, activation='relu'),
    Dense(units=25, activation='relu'),


])

#then we will define the cost function and the loss function
# logisctic loss is also know as the binary_crossentropy function in tensorflow
# if it is regression based, then we can use the mean squared error for the loss function
model.compile(loss=BinaryCrossentropy())

#then we will use the gradienct decnet algorithm
#there are many activation functions such
#linear
#sigmoid 1/1+e^-z
#relu if z less than 0 than 0, else equal toz, rectifier linear unit
model.fit(X,y,epochs=100)

#for the binary classfication e generally use sigmoid function as the activation, for instacnce true or false
#for a regression based problem we generally use linear based activation function
#and if you predicting the price of somthing that can never be negative, then you should go fpor relu activation
# relu is the most common hidden layer activation function and faster to compute, faster learning as
#all of the layers hgsoudl not be linear since that causes it to give out just loinear values
#do not use linear funtion (recommended) isntead use relu activation inside the hidden layer insre