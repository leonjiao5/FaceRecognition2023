import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Dense, Flatten

import tkinter as tk
from PIL import Image, ImageTk

class L1Dist(Layer):
    # Inheritance of the Layer class
    def __init__(self, **kwargs):
        super().__init__()

    # Calculate distance (similarity)
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    # Resize image to 100x100x3 - necessary for the neural network because shapes are standardized
    img = tf.image.resize(img, (100, 100))
    # Scale image by dividing every value by 255
    img /= 255.0

    return img


def verify(frame, model, detection_threshold, verification_threshold):
    cv2.imwrite(os.path.join("application_data", "input_image", "input_image.jpg"), frame)
    results = []
    for image in os.listdir(os.path.join("application_data", "verification_images")):
        input_img = preprocess(os.path.join("application_data", "input_image", "input_image.jpg"))
        validation_img = preprocess(os.path.join("application_data", "verification_images", image))

        # Make Predictions
        result = model(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Calculate the proportion of predicted results that are positive
    detection_total = np.sum(np.array(results) > detection_threshold)
    verification = detection_total / len(os.listdir(os.path.join("application_data", "verification_images")))
    verdict = verification > verification_threshold

    status.configure(text="Verified" if verdict else "Unverified")
    print(np.squeeze(results))
    print("Verified:", verdict, "\n")
    return results, verdict


def pos(e):
    x = e.x
    y = e.y
    print(x, y)


if __name__ == "__main__":
    # Load the Neural Network we will be using
    model = tf.keras.models.load_model("siamesemodelv2.h5",
                                    custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":tf.losses.BinaryCrossentropy})

    # Initialize the main window
    main = tk.Tk()
    main.title("Face Verification")
    main.geometry("1000x600")

    video = cv2.VideoCapture(1)
    frame = video.read()[1]

    # Add widgets
    button = tk.Button(main)
    pixel = tk.PhotoImage(width=1, height=1)
    button.config(text="Verify", font="Titillium_Web 18", width=600, height=50, background="#CFCFCF",
                  image=pixel, compound="c", command=lambda: verify(frame, model, 0.5, 0.5))
    button.place(x=50, y=450)

    status = tk.Label(main, text="Status: Uninitiated", font="Titillium_Web 15")
    status.place(x=355, y=540, anchor=tk.CENTER)

    video_label = tk.Label(main)
    video_label.place(x=225, y=100)

    # Display webcam feed
    while True:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = frame[115:115+250, 190:190+250, :]

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(cv2image))

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        main.update()



    # Run
    # main.bind("<Motion>", pos)
    # main.mainloop()
