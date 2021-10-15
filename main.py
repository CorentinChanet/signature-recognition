from src.preprocessing import enhance_image, resize_image
from src.data_loader import load_image
from src.extraction import Component, ComponentParser

import os
import joblib

# parser = joblib.load("parsed_documents/8e20c5b100299349efd0339019392688_2.z")
# image = load_image("data/train/8e20c5b100299349efd0339019392688_2.tif")


image = resize_image(enhance_image(load_image("data/train/00ba5cc657c8c203c4ed5e339f7d50e9.tif")))
parser = ComponentParser("00ba5cc657c8c203c4ed5e339f7d50e9")
parser.parse(image)

parser.output(resize_image(load_image("data/train/00ba5cc657c8c203c4ed5e339f7d50e9.tif")))