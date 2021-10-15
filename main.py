from src.preprocessing import load_image, enhance_image
from src.extraction import Component, ComponentParser

img = load_image('./data/train/af5cfb0ee6d4caa263f332da79d907a7.tif')

enhanced_img = enhance_image(img)

parser = ComponentParser('af5cfb0ee6d4caa263f332da79d907a7')
parser.parse(enhanced_img)
parser.output(img)