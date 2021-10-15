from preprocessing import enhance_image, resize_image
from data_loader import load_image
from extraction import Component, ComponentParser

import os
import joblib

parsed_documents = []

for root, folders, files in os.walk("../data/train"):
    for idx, file_name in enumerate(files):
        img_path = os.path.join(os.path.normpath('../data/train'), file_name)
        img = load_image(img_path)

        enhanced_img = enhance_image(img)
        resized_img = resize_image(enhanced_img)

        parser = ComponentParser(file_name[:-4])
        parser.parse(resized_img)
        parsed_documents.append(parser)
        print(f"Finished parsing document {idx}/1039 : {file_name[:-4]}")
        joblib.dump(parser, os.path.join('../parsed_documents', file_name[:-4]) +".z", compress=True)