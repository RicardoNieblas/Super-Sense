import warnings
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

warnings.filterwarnings("ignore")

# Load the pre-trained model
model = load_model("best_model.h5")

# Load Haar Cascade for face detection
faceHaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades \
                                        + "haarcascade_frontalface_default.xml")

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, testImage = cap.read()

    # If frame capture fails, continue to the next iteration
    if not ret:
        continue

    # Convert the captured image to grayscale
    grayImage = cv2.cvtColor(src = testImage,
                             code = cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale image
    facesDetected = faceHaarCascade.detectMultiScale(image = grayImage,
                                                     scaleFactor = 1.32,
                                                     minNeighbors = 5)

    # Loop through the detected faces
    for (x, y, w, h) in facesDetected:
        # Draw a rectangle around the detected face
        cv2.rectangle(img = testImage,
                      pt1 = (x, y),
                      pt2 = (x + w, y + h),
                      color = (255, 0, 0),
                      thickness = 7)

        # Crop and preprocess the region of interest (face)
        regionOfInterestGray = grayImage[y:y + h, x:x + w]
        regionOfInterestGray = cv2.resize(src = regionOfInterestGray,
                                          dsize = (224, 224))
        imagePixels = image.img_to_array(img = regionOfInterestGray)
        imagePixels = np.expand_dims(a = imagePixels,
                                     axis = 0)
        imagePixels /= 255

        # Make predictions using the loaded model
        predictions = model.predict(imagePixels)

        # Find the emotion label with the highest prediction score
        maxIndex = np.argmax(predictions[0])
        emotions = ('angry',
                    'disgust',
                    'fear',
                    'happy',
                    'sad',
                    'surprise',
                    'neutral')
        predictedEmotion = emotions[maxIndex]

        # Display the predicted emotion on the image
        cv2.putText(img = testImage,
                    text = predictedEmotion,
                    org = (int(x), int(y)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color = (0, 0, 255),
                    thickness = 2)

    # Resize the image for display
    resizedImage = cv2.resize(src = testImage,
                              dsize = (1000, 700))

    # Show the image with predicted emotions
    cv2.imshow(winname = "Super Sense",
               mat = resizedImage)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(10) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()