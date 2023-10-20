import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from detectText import analyzeSentiment

def process_message():
    input_text = input_text_entry.get()
    # Call your function to process the input text and get a response
    sentiment, confidence = analyzeSentiment(input_text)
    conversation_text.config(state='normal')
    conversation_text.insert('end', f'You: {input_text}\n')
    conversation_text.insert('end', f"Bot: It's a {sentiment} sentiment, with {confidence} confidence.\n")
    conversation_text.config(state='disabled')
    input_text_entry.delete(0, 'end')

# Load the pre-trained model
model = load_model("best_model.h5")
faceHaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

def display_camera():
    def process_frame():
        while True:
            ret, testImage = cap.read()
            if not ret:
                continue
            grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
            facesDetected = faceHaarCascade.detectMultiScale(grayImage, scaleFactor=1.32, minNeighbors=5)
            for (x, y, w, h) in facesDetected:
                cv2.rectangle(testImage, (x, y), (x + w, y + h), (255, 0, 0), 7)
                regionOfInterestGray = grayImage[y:y + h, x:x + w]
                regionOfInterestGray = cv2.resize(regionOfInterestGray, (224, 224))
                imagePixels = image.img_to_array(regionOfInterestGray)
                imagePixels = np.expand_dims(imagePixels, axis=0)
                imagePixels /= 255
                predictions = model.predict(imagePixels)
                maxIndex = np.argmax(predictions[0])
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predictedEmotion = emotions[maxIndex]
                cv2.putText(testImage, predictedEmotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            resizedImage = cv2.resize(testImage, (1000, 700))
            cv2.imshow("Super Sense", resizedImage)
            if cv2.waitKey(10) == ord('q'):
                break
    process_frame()
    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Messaging and Camera App")
root.geometry("800x600")

# Create notebook (tabbed interface)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand='yes')

# Messaging tab
messaging_frame = ttk.Frame(notebook)
notebook.add(messaging_frame, text='Messaging')

conversation_text = tk.Text(messaging_frame, wrap='word', state='disabled')
conversation_text.pack(fill='both', expand='yes')

input_text_entry = tk.Entry(messaging_frame)
input_text_entry.pack(fill='both', expand='yes')

send_button = tk.Button(messaging_frame, text='Send', command=process_message)
send_button.pack()

# Camera tab
camera_frame = ttk.Frame(notebook)
notebook.add(camera_frame, text='Camera')

camera_label = tk.Label(camera_frame, text='Camera feed will be shown here:')
camera_label.pack()

start_camera_button = tk.Button(camera_frame, text='Start Camera', command=display_camera)
start_camera_button.pack()

root.mainloop()