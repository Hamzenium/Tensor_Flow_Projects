#if there are more than two onjects to be found, then we have multi class classification
#mutliclass calssification can learn about differnt boundries of didfferent set of values
#softmax regression is an activation function that can take on values more than 2, and can hold values N numbers
import tensorflow as tf
#first we will specify the model we will be using
from tensorflow.keras.layers import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(units=25,activation='relu'),
    Dense(units=15,activation='relu'),
    Dense(units=10,activation='softmax')
])
#this is the cost function we used for the
from tensorflow.keras.losses import SparseCategoricalCrossEntropy
model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True)
#train on data to mnimize the cost error
#FROM_LOGITS=TRUE IS MORE numericall accurate and givs better result
#log
model.fit(X,y,epochs=100)
f_x = tf.nn.sigmoid(logit)
#there is another algorithm called ath e adam algortithm which if the gradient algorthim goes in the same dorection
#then the adam algorithm causes the learning rate to be increesed
# and if the path is going right and side ways this alogotiythm can also decrease the learning rate
#model.compile(optimzers=tf.keras.optimizers.Adam(learning_rate=1e-3)
 #convolutuional layer is a layer that looks at only part of the previous neuron 