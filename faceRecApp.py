from pathlib import Path
import face_recognition as fr
import pickle
from collections import Counter
from PIL import Image, ImageDraw, ImageTk

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import os
import uuid

ENCODINGS_PATH = Path("application_data/encodings.pkl")
NEW_ENCODINGS_PATH = Path("application_data/new_encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


def encode_known_faces(model: str="hog", encodings_location: Path=ENCODINGS_PATH) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = fr.load_image_file(filepath)

        face_locations = fr.face_locations(image, model=model)
        face_encodings = fr.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


# encode_known_faces()
# recognize_faces("unknown.webp")


name = "Unknown"
def recognize_faces(image_location: str, model: str="hog") -> None:
    input_image = fr.load_image_file(image_location)
    input_locations = fr.face_locations(input_image, model=model)

    if not input_locations:
        status.configure(text="No face was detected!")
        return

    input_encodings = fr.face_encodings(input_image, input_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, test_encoding in zip(input_locations, input_encodings):
        name = _recognize_face(test_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
            status.configure(text="UNVERIFIED: User unknown!")
        else:
            status.configure(text="VERIFIED: Welcome user {}!".format(name))
        # print(name, bounding_box)
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()

def _recognize_face(test_encoding, loaded_encodings):
    boolean_matches = fr.compare_faces(loaded_encodings["encodings"], test_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((text_left, text_top), name, fill=TEXT_COLOR)

def verify_face(model: str="hog"):
    file_path = os.path.join("application_data", "test_images", "test_image0.png")
    cv2.imwrite(file_path, frame)
    recognize_faces(file_path, model)


def enable_live():
    live[0] = not live[0]
    text = "ON" if live[0] else "OFF"
    live_mode.configure(text="Live: {}".format(text))

def live_recognition(frame):
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = np.asarray(input_image)

    face_locations = fr.face_locations(input_image, model="hog")
    face_encodings = fr.face_encodings(input_image, face_locations)

    for bounding_box, test_encoding in zip(face_locations, face_encodings):
        name = _recognize_face(test_encoding, loaded_encodings)
        if not name:
            name = "Unknown"

        top, right, bottom, left = bounding_box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom-25), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+5, bottom-5), font, 0.5, (255, 255, 255), 1)

'''
Functions to add a user to the verified list, connecting to the camera
'''
def add_user():
    warning_label.config(text="", font="Helvetica 16")
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

    submit_button = tk.Button(entry_box, text="Submit", command=lambda: submit(entry_box, entry1, entry2))
    submit_button.grid(row=2, column=1)

def submit(entry_box, entry1, entry2):
    first = entry1.get()
    last = entry2.get()
    name = first + "_" + last

    if first == "" and last == "":
        warning_label.config(text="ERROR: Username is blank.")
        warning_label.place(x=500, y=50, anchor=tk.CENTER)
        return
    elif first == "" or last == "":
        name = first + last

    Path("application_data/users/{}".format(name)).mkdir(exist_ok=True)
    path_name = os.path.join("application_data", "users", name)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = np.asarray(rgb)

    face_location = fr.face_locations(input_image, model="hog")
    face_encoding = fr.face_encodings(input_image, face_location)[0]
    top, right, bottom, left = face_location[0]
    face_image = input_image[top:bottom, left:right, :]

    cv2.imwrite(os.path.join(path_name, "{}.jpg".format(uuid.uuid1())), face_image)

    usernames.add(name)
    loaded_encodings["names"].append(name)
    loaded_encodings["encodings"].append(face_encoding)

    with NEW_ENCODINGS_PATH.open(mode="wb") as f:
        pickle.dump(loaded_encodings, f)

    entry_box.destroy()


'''
Functions to remove a user from the verified list
'''
def remove_user():
    warning_label.config(text="", font="Helvetica 16")
    entry_box = tk.Tk()
    entry_box.title("Remove User")

    label1 = tk.Label(entry_box, text="First Name")
    label1.grid(row=0, column=0)
    label2 = tk.Label(entry_box, text="Last Name")
    label2.grid(row=1, column=0)

    entry1 = tk.Entry(entry_box)
    entry1.grid(row=0, column=1)
    entry2 = tk.Entry(entry_box)
    entry2.grid(row=1, column=1)

    submit_button = tk.Button(entry_box, text="Submit", command=lambda: submit2(entry_box, entry1, entry2))
    submit_button.grid(row=2, column=1)

def submit2(entry_box, entry1, entry2):
    first = entry1.get()
    last = entry2.get()
    name = first + "_" + last

    if first == "" and last == "":
        warning_label.config(text="ERROR: Username is blank.")
        warning_label.place(x=500, y=50, anchor=tk.CENTER)
        return
    elif first == "" or last == "":
        name = first + last

    if name in usernames:
        usernames.remove(name)
        index = loaded_encodings["names"].index(name)
        loaded_encodings["names"].pop(index)
        loaded_encodings["encodings"].pop(index)

        with NEW_ENCODINGS_PATH.open(mode="wb") as f:
            pickle.dump(loaded_encodings, f)

    else:
        warning_label.config(text="ERROR: User does not exist and was not removed.")
        warning_label.place(x=500, y=50, anchor=tk.CENTER)

    entry_box.destroy()


'''
Lists all verified users
'''
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

    for index, user in enumerate(usernames):
        # image_path = os.path.join("application_data", "verification_images", user)
        name_label = tk.Label(scroll_frame, text=user, font="Helvetica 14")
        name_label.pack(anchor=tk.CENTER)
        # name_label.place(x=(index%4)*300+175, y=(index//4)*350+300, anchor=tk.CENTER)

    container.pack()
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    scroll_frame.pack()

    users_window.update()


'''
Main method where important code is executed
'''
if __name__ == "__main__":
    # Initialize verified users
    with NEW_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    usernames = set(loaded_encodings["names"])

    # Initialize the main window
    main = tk.Tk()
    main.title("Face Verification")
    main.geometry("1000x600")

    video = cv2.VideoCapture(1)
    frame = video.read()[1]

    # Add widgets
    button = tk.Button(main)
    pixel = tk.PhotoImage(width=1, height=1)
    button.config(text="Verify", font="Titillium_Web 18", width=450, height=50, background="#CFCFCF",
                  image=pixel, compound="c", command=lambda: verify_face(frame))
    button.place(x=50, y=450)

    live = [False]
    live_mode = tk.Button(main)
    live_mode.config(text="Live: OFF", font="Titillium_Web 14", width=150, height=50, background="#CFCFCF",
                  image=pixel, compound="c", command=enable_live)
    live_mode.place(x=500, y=450)

    status = tk.Label(main, text="Status: Uninitiated", font="Titillium_Web 15")
    status.place(x=355, y=540, anchor=tk.CENTER)

    video_label = tk.Label(main)
    video_label.place(x=100, y=100)

    add = tk.Button(main)
    add.config(text="Add User", font="Titillium_Web 15", width=250, height=120, background="#E0E0E0",
                    image=pixel, compound="c", command=add_user)
    add.place(x=700, y=100)

    remove = tk.Button(main)
    remove.config(text="Remove User", font="Titillium_Web 15", width=250, height=120, background="#E0E0E0",
                    image=pixel, compound="c", command=remove_user)
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

        # Resize the frame - 400x300
        frame = frame[120:420, 70:570, :]

        if live[0]:
            status.configure(text="Status: LIVE")
            live_recognition(frame)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(cv2image))

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        main.update()