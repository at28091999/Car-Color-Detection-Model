import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import filedialog, Tk, Label, Button, Canvas, PhotoImage
from PIL import Image, ImageTk

color_model = load_model("car_color_model.h5")
color_labels = ['black', 'blue', 'gray', 'green', 'red', 'white', 'yellow']

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

root = Tk()
root.title("Car Color Detection with People Count")

canvas = Canvas(root, width=800, height=600)
canvas.pack()

label_result = Label(root, text="", font=("Helvetica", 14))
label_result.pack()

def detect_objects(img_path):
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.forward(layer_names)

    car_boxes = []
    person_count = 0

    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if label == "car":
                    car_boxes.append((x, y, int(w), int(h)))
                elif label == "person":
                    person_count += 1

    for (x, y, w, h) in car_boxes:
        car_crop = image[y:y+h, x:x+w]
        if car_crop.size == 0: continue

        car_resized = cv2.resize(car_crop, (128, 128))
        car_input = img_to_array(car_resized) / 255.0
        car_input = np.expand_dims(car_input, axis=0)

        color_pred = color_model.predict(car_input)
        color_idx = np.argmax(color_pred)
        color_name = color_labels[color_idx]

        box_color = (0, 0, 255) if color_name == 'blue' else (255, 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(image, color_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    label_result.config(text=f"Cars Detected: {len(car_boxes)} | People: {person_count}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pil_img = pil_img.resize((800, 600))
    imgtk = ImageTk.PhotoImage(pil_img)
    canvas.create_image(0, 0, anchor='nw', image=imgtk)
    canvas.imgtk = imgtk

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_objects(file_path)

Button(root, text="Upload Image", command=upload_image).pack()

root.mainloop()
