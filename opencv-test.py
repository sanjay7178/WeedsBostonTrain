import cv2
import numpy as np
from tensorflow.keras.models import load_model

# set the path to the model file
model_path = 'vgg16_epochs_25.h5'

# load the model
model = load_model(model_path)

# set the input image size
input_size = (224, 224)

# set the input mean and standard deviation for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# set the input scale factor
scale_factor = 1/255.0

# set the labels for the classes
class_labels = ['CELOSIA ARGENTEA L',  'CROWFOOT GRASS'	, 'PURPLE CHLORIS']

# set the index of the camera to use
camera_index = 0

# create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(camera_index)

# loop over the frames from the camera
while True:
    # read a frame from the camera
    ret, frame = cap.read()
    
    # resize the frame to the input size
    frame = cv2.resize(frame, input_size)
    
    # convert the frame to a numpy array and normalize it
    image = np.asarray(frame, dtype=np.float32)
    image /= 255.0
    image -= mean
    image /= std
    
    # add a batch dimension to the input image
    image = np.expand_dims(image, axis=0)
    
    # run the inference on the input image
    prediction = model.predict(image)
    
    # get the predicted class label
    predicted_class = np.argmax(prediction)
    
    # get the predicted class probability
    predicted_prob = prediction[0][predicted_class]
    
    # draw the predicted class label and probability on the frame
    label = '{}: {:.2f}%'.format(class_labels[predicted_class], predicted_prob*100)
    cv2.putText(frame, label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # show the frame
    cv2.imshow('Camera', frame)
    
    # wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
