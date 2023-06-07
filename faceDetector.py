from pathlib import Path
import face_recognition as fr
import pickle
from collections import Counter
from PIL import Image, ImageDraw

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Path("training").mkdir(exist_ok=True)
# Path("output").mkdir(exist_ok=True)
# Path("validation").mkdir(exist_ok=True)

def encode_known_faces(model: str="hog", encodings_location: Path=DEFAULT_ENCODINGS_PATH) -> None:
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

def recognize_faces(image_location: str, model: str="hog", encodings_location: Path=DEFAULT_ENCODINGS_PATH) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = fr.load_image_file(image_location)
    input_locations = fr.face_locations(input_image, model=model)
    input_encodings = fr.face_encodings(input_image, input_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, test_encoding in zip(input_locations, input_encodings):
        name = _recognize_face(test_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
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



def validate(model: str="hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(image_location=str(filepath.absolute()), model=model)


encode_known_faces()
# recognize_faces("unknown.webp")
# validate()
