from src.preprocessing import enhance_image
from src.data_loader import load_image
from src.extraction import Component, ComponentParser

import os
import joblib

# parser = joblib.load("parsed_documents/8e20c5b100299349efd0339019392688_2.z")
# image = load_image("data/train/8e20c5b100299349efd0339019392688_2.tif")


image = enhance_image(load_image("data/train/8e20c5b100299349efd0339019392688_2.tif"))
parser = ComponentParser("8e20c5b100299349efd0339019392688_2")
parser.parse(image)

parser.output(load_image("data/train/8e20c5b100299349efd0339019392688_2.tif"))