
# -Object-detection-using-web-camera
# YOLOv8 Object Detection Using Laptop Camera
# Name : Ayshwariya J
# REG NO:212224230030

## Aim

To access the **laptop camera**, capture an image, and detect objects using **YOLOv8**.

## Requirements

* Anaconda
* Jupyter Notebook
* Laptop Camera

## Steps

1. Open **Jupyter Notebook** using Anaconda.
2. Access your **laptop camera** using OpenCV.
3. Wait for **5 seconds** and capture an image.
4. Display the captured image.
5. Apply **YOLOv8 object detection**.
6. Display the detected image with **bounding boxes and labels**.

## Algorithm

```text
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```

## Student Task

* Capture your own image using the laptop camera.
* Detect the objects present in the image.
* Display the final output.
* Perform the experiment with **3 different scenes**.

## Submission

* Jupyter Notebook (`.ipynb`)
* Original captured images
* YOLOv8 output images
* Screenshot of the final result

## GitHub Reference

https://github.com/ultralytics/ultralytics

**Platform:** Anaconda + Jupyter Notebook only.

## Program
```
from ultralytics import YOLO
import cv2
import time
from IPython.display import display, clear_output, Image

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Object Detection Started!")

start_time = time.time()

try:
    while time.time() - start_time < 30:

        ret, frame = cap.read()

        if not ret:
            print("Frame not captured")
            break

        # Detect objects
        results = model.predict(frame, conf=0.20, verbose=False)

        # Draw bounding boxes and labels
        output = results[0].plot()

        # Display output in Jupyter
        _, buffer = cv2.imencode(".jpg", output)

        clear_output(wait=True)
        display(Image(data=buffer.tobytes()))

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Object Detection Stopped!")

```
## Output
<img width="982" height="691" alt="image" src="https://github.com/user-attachments/assets/050a5b01-44b1-4f4d-814a-29f97a86d5f9" />


