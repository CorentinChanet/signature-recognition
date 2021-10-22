from src.preprocessing import enhance_image, resize_image
from src.data_loader import load_image
from src.extraction import Component, ComponentParser

import os
import joblib

# parser = joblib.load("parsed_documents/8e20c5b100299349efd0339019392688_2.z")
# image = load_image("data/train/8e20c5b100299349efd0339019392688_2.tif")


# image = resize_image(enhance_image(load_image("data/train/0a948131fe85c38152c0b9b22f7c09fc_4.tif")))
# parser = ComponentParser("0a948131fe85c38152c0b9b22f7c09fc_4")
# parser.parse(image)
#
# parser.output(resize_image(load_image("data/train/0a948131fe85c38152c0b9b22f7c09fc_4.tif")))


from src.preprocessing import enhance_image, resize_image
from src.data_loader import load_image
from src.extraction import Component, ComponentParser
from src.CNN import try_model

import os
import joblib
import cv2


def run_parsing():
    parsed_documents = {}
    for root, folders, files in os.walk("data/test"):
        for idx, file_name in enumerate(files):
            img_path = os.path.join(os.path.normpath('data/test'), file_name)
            img = load_image(img_path)

            enhanced_img = enhance_image(img)
            resized_img = resize_image(enhanced_img, dim=(595*2, 842*2))

            parser = ComponentParser(file_name[:-4])
            parser.parse(resized_img)

            parser.output(resize_image(img, dim=(595*2, 842*2)), mode='write')
            parsed_documents[file_name[:-4]] = parser
            print(f"Finished parsing document {idx}/{len(files)} : {file_name[:-4]}")

    return parsed_documents

def run_predictions():
    pass


if __name__ == "__main__":
    parsed_documents = run_parsing()
