import cv2

with open("../data/in2.jpg") as f:
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
	    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    thresh, 8, cv2.CV_32S) # The integer is the connectivity type (4 or 8)


