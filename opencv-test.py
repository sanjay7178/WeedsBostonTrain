import cv2
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Open the camera feed
cap = cv2.VideoCapture(0)

# Loop over frames from the camera feed
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the input image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    # Pass the preprocessed image through the model to obtain a prediction
    pred = model.predict(img)

    # Postprocess the prediction to obtain a final output
    # In this example, we assume the model predicts a single class label
    label = tf.argmax(pred, axis=1)
    label = label.numpy()[0]

    # Display the output on the camera feed
    cv2.putText(frame, f'Class: {label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
