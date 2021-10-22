from PIL import Image
import os
import shutil
import cv2
import numpy as np

def enhance_image(image : np.ndarray) -> np.ndarray:
    '''Docstring'''

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]

    return thresh

# function to resize image
def resize_image(src_image, size=(256, 256), bg_color="white"):
    from PIL import Image, ImageOps

    #resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)

    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)

    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))

    new_image = np.array(new_image)
    # return the resized image
    new_image = Image.fromarray(new_image)
    #new_image = src_image.resize((256,256), Image.BILINEAR)

    return new_image

training_folder_name = '/Users/Corty/Sync/becode_projects/Python/signature-recognition/parsed_documents'

# New location for the resized images
train_folder = '/Users/Corty/Sync/becode_projects/Python/signature-recognition/parsed_documents_CNN'


# Create resized copies of all of the source images
size = (256,256)

# Create the output folder if it doesn't already exist
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)

# Loop through each subfolder in the input folder
print('Transforming images...')
for root, folders, files in os.walk(training_folder_name):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        # Create a matching subfolder in the output dir
        saveFolder = os.path.join(train_folder,sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        # Loop through the files in the subfolder
        file_names = os.listdir(os.path.join(root,sub_folder))
        for file_name in file_names:
            if '.DS_Store' in file_name:
                continue
            # Open the file
            file_path = os.path.join(root,sub_folder, file_name)
            #print("reading " + file_path)
            image = Image.open(file_path)
            # Create a resized version and save it
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            #print("writing " + saveAs)
            resized_image.save(saveAs)

print('Done.')