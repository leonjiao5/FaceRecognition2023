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
from tkinter import ttk
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
    status.configure(text="Processing...")
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


users = ["Leon_Jiao"]
def change_user(add=True):
    warning_label.config(text="")
    entry_box = tk.Tk()
    entry_box.title("Add User")

    label1 = tk.Label(entry_box, text="First Name")
    label1.grid(row=0, column=0)
    label2 = tk.Label(entry_box, text="Last Name")
    label2.grid(row=1, column=0)

    entry1 = tk.Entry(entry_box)
    entry1.grid(row=0, column=1)
    entry2 = tk.Entry(entry_box)
    entry2.grid(row=1, column=1)

    submit_button = tk.Button(entry_box, text="Submit", command=lambda: submit(entry_box, entry1, entry2, add))
    submit_button.grid(row=2, column=1)


def submit(entry_box, entry1, entry2, add):
    first = entry1.get()
    last = entry2.get()
    name = first + "_" + last

    if add:
        users.append(name)
    else:
        if name in users:
            users.remove(name)
        else:
            warning_label.config(text="ERROR: User does not exist and was not removed.", font="Helvetica 16")
            warning_label.place(x=500, y=50, anchor=tk.CENTER)

    entry_box.destroy()


def list_users():
    users_window = tk.Tk()
    users_window.title("Verified Users")
    # users_window.geometry("1250x800")

    container = ttk.Frame(users_window)
    canvas = tk.Canvas(container)
    scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

    scroll_frame = ttk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)

    for index, user in enumerate(users):
        # image_path = os.path.join("application_data", "verification_images", user)
        name_label = tk.Label(scroll_frame, text=user, font="Helvetica 14")
        name_label.pack(anchor=tk.CENTER)
        # name_label.place(x=(index%4)*300+175, y=(index//4)*350+300, anchor=tk.CENTER)

    container.pack()
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    scroll_frame.pack()

    users_window.update()

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
                  image=pixel, compound="c", command=lambda: verify(frame, model, 0.8, 0.8))
    button.place(x=50, y=450)

    status = tk.Label(main, text="Status: Uninitiated", font="Titillium_Web 15")
    status.place(x=355, y=540, anchor=tk.CENTER)

    video_label = tk.Label(main)
    video_label.place(x=225, y=100)

    add = tk.Button(main)
    add.config(text="Add User", font="Titillium_Web 15", width=250, height=120, background="#E0E0E0",
                    image=pixel, compound="c", command=change_user)
    add.place(x=700, y=100)

    remove = tk.Button(main)
    remove.config(text="Remove User", font="Titillium_Web 15", width=250, height=120, background="#E0E0E0",
                    image=pixel, compound="c", command=lambda: change_user(False))
    remove.place(x=700, y=240)

    list_user = tk.Button(main)
    list_user.config(text="List Verified Users", font="Titillium_Web 15", width=250, height=120, background="#E0E0E0",
                       image=pixel, compound="c", command=list_users)
    list_user.place(x=700, y=380)

    warning_label = tk.Label(main)

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
