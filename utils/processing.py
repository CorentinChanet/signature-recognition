import cv2
import numpy as np
import matplotlib.pyplot as plt

class Component:
    def __init__(self, component_mask : np.ndarray, centroid : dict, bbox : dict):
        self.component_mask = component_mask
        self.bbox = bbox
        self.centroid = centroid

class ComponentParser:
    def __init__(self, image_id):
        self.components = []
        self.image_id = image_id

    def __str__(self):
        return self.image_id

    @staticmethod
    def _proximity_check(right_edge, left_edge, cY_1, cY_2, threshold=50):
        if (abs(right_edge - left_edge) < threshold) and (abs(cY_1 - cY_2) < threshold):
            return True
        else:
            return False

    def parse(self, enhanced_image : np.ndarray) :
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced_image)

        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            (cX, cY) = centroids[i]
            keepWidth = w > 85 and w < 2000
            keepHeight = h > 85 and h < 3000
            keepArea = area > 1000 and area < 12000

            if all((keepWidth, keepHeight, keepArea)):
                componentMask = (labels == i).astype("uint8") * 255
                centroid = {'x' : cX,
                            'y' : cY}
                bbox = {'x' : x,
                        'y' : y,
                        'w' : w,
                        'h' : h,
                        'xw' : x + w,
                        'yh' : y + h}

                new_component = Component(componentMask, centroid, bbox)

                # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

                idx = 0

                while idx < len(self.components):

                    proximity = False

                    if self._proximity_check(self.components[idx].bbox['xw'], new_component.bbox['x'],
                                             self.components[idx].centroid['y'], new_component.centroid['y']):
                        left, right = self.components[idx], new_component
                        proximity = True

                    elif self._proximity_check(new_component.bbox['xw'], self.components[idx].bbox['x'],
                                               new_component.centroid['y'], self.components[idx].centroid['y']):
                        left, right = new_component, self.components[idx]
                        proximity = True

                    if proximity:
                        merged_component_mask = left.component_mask + right.component_mask
                        merged_centroid = {'x': (left.centroid['x'] + right.centroid['x']) / 2,
                                           'y': (left.centroid['y'] + right.centroid['y']) / 2}
                        merged_bbox = {'x': left.bbox['x'],
                                       'y': min(left.bbox['y'], right.bbox['y']),
                                       'w': right.bbox['xw'] - left.bbox['x'],
                                       'h': max(left.bbox['yh'], right.bbox['yh']) - min(
                                           left.bbox['y'], right.bbox['y']),
                                       'xw': right.bbox['xw'],
                                       'yh': max(left.bbox['yh'], right.bbox['yh'])}

                        merged_component = Component(merged_component_mask, merged_centroid, merged_bbox)

                        self.components.pop(idx)
                        new_component = merged_component
                        idx = 0

                    if not proximity:
                        idx += 1

                self.components.append(new_component)

    def output(self):
        if not self.components:
            return "No signatures were found on this document"
        else:
            output = image.copy()
            for component in self.components:
                x = component.bbox['x']
                y = component.bbox['y']
                xw = component.bbox['xw']
                yh = component.bbox['yh']
                cv2.rectangle(output, (x, y), (xw, yh), (0, 255, 0), 3)
            cv2.imshow("output", output)
            cv2.waitKey(0)

image = plt.imread("../data/train/0d178d095434170eac2cb58cc244bb8c_2.tif")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3, 3),np.uint8)
erosion = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations = 2)

thresh = cv2.threshold(erosion, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# cv2.imshow('check', thresh)
# cv2.waitKey(0)

parser = ComponentParser('0d178d095434170eac2cb58cc244bb8c_2')
parser.parse(thresh)
parser.output()





