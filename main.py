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

import os
import joblib
import cv2

parsed_documents = {}

for root, folders, files in os.walk("data/train"):
    for idx, file_name in enumerate(files):
        img_path = os.path.join(os.path.normpath('data/train'), file_name)
        img = load_image(img_path)

        enhanced_img = enhance_image(img)
        resized_img = resize_image(enhanced_img)

        parser = ComponentParser(file_name[:-4])
        parser.parse(resized_img)
        parser.generate_rois(resized_img)
        parsed_documents[file_name[:-4]] = parser
        print(f"Finished parsing document {idx}/1039 : {file_name[:-4]}")

