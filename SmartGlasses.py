# Smart Assistive Glasses Using Lightweight AI Models for Real-Time Object Identification,Navigation, and Enhanced Visual Perception for the Visually Impaired 
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import os
import tflite_runtime.interpreter as tflite

# GPIO Setup for Ultrasonic Sensor
TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Button GPIO Pins
BUTTON1 = 17  # Detect Objects
BUTTON2 = 27  # (Unused now)
BUTTON3 = 22  # Navigation
BUTTON4 = 5   # Exit

GPIO.setup(BUTTON1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON2, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Optional, can remove
GPIO.setup(BUTTON3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON4, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="ssd_mobilenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("coco_labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
if class_names[0].lower() != "n/a" and len(class_names) == 90:
    class_names = ["N/A"] + class_names

# Capture frame from camera
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Error: Failed to capture frame")
        return None
    return frame

# Object Detection with Audio
def object_detection():
    frame = capture_image()
    if frame is None:
        os.system('espeak "No image captured"')
        return "No image captured"

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    if list(input_data.shape) != list(input_shape):
        return f"Shape mismatch: model expects {input_shape}, got {input_data.shape}"

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detected_objects = []
    for i in range(len(scores)):
        if scores[i] > 0.2:
            class_id = int(classes[i])
            label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            print(f"Detected: {label} (score={scores[i]:.2f})")
            detected_objects.append(label)

    if detected_objects:
        spoken_text = "I see " + ", ".join(set(detected_objects))
        os.system(f'espeak "{spoken_text}"')
        return spoken_text
    else:
        os.system('espeak "No objects detected"')
        return "No objects detected"

# Obstacle Detection with Audio
def detect_obstacles():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 1
    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        start_time = time.time()
    if time.time() >= timeout:
        os.system('espeak "Ultrasonic timeout"')
        return "Ultrasonic timeout"

    timeout = time.time() + 1
    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        stop_time = time.time()
    if time.time() >= timeout:
        os.system('espeak "Ultrasonic timeout"')
        return "Ultrasonic timeout"

    distance = (stop_time - start_time) * 34300 / 2
    distance = round(distance, 2)
    print(f"Distance: {distance} cm")

    if distance < 50:
        os.system('espeak "Obstacle ahead"')

    return distance

# Main Menu
def main():
    print("Smart Glasses System Ready.")
    print("Press the RUN button to begin...")

    # Wait for RUN (BUTTON2) to be pressed to start system
    while GPIO.input(BUTTON2) == GPIO.HIGH:
        time.sleep(0.1)

    print("RUN button pressed. System is active.")

    try:
        while True:
            if GPIO.input(BUTTON1) == GPIO.LOW:
                print("Detecting objects...")
                print(object_detection())
                time.sleep(1)

            elif GPIO.input(BUTTON3) == GPIO.LOW:
                print("Checking for obstacles...")
                distance = detect_obstacles()
                if isinstance(distance, float):
                    print(f"Distance ahead: {distance:.2f} cm")
                else:
                    print(distance)
                time.sleep(1)

            elif GPIO.input(BUTTON4) == GPIO.LOW:
                os.system('espeak "Exiting system"')
                print("Exiting...")
                break

            time.sleep(0.1)

    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
