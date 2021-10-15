import cv2
import numpy as np

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
    def _proximity_check(right_edge : int, left_edge : int, cY_1 : int, cY_2 : int,
                         threshold_x=100, threshold_y=55) -> bool:
        '''Docstring'''

        if (abs(right_edge - left_edge) < threshold_x) and (abs(cY_1 - cY_2) < threshold_y):
            return True
        else:
            return False

    def parse(self, enhanced_image : np.ndarray) :
        '''Docstring'''

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced_image)

        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            (cX, cY) = centroids[i]
            keepWidth = w > 30 and w < 1500
            keepHeight = h > 30 and h < 1000
            keepArea = area > 200 and area < 35000

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
                        merged_bbox = {'x': min(left.bbox['x'], right.bbox['x']),
                                       'y': min(left.bbox['y'], right.bbox['y']),
                                       'w': max(left.bbox['xw'], right.bbox['xw']) - min(
                                           left.bbox['x'], right.bbox['x']),
                                       'h': max(left.bbox['yh'], right.bbox['yh']) - min(
                                           left.bbox['y'], right.bbox['y'])}

                        merged_bbox['xw'] = merged_bbox['x'] + merged_bbox['w']
                        merged_bbox['yh'] = merged_bbox['y'] + merged_bbox['h']

                        merged_component = Component(merged_component_mask, merged_centroid, merged_bbox)

                        self.components.pop(idx)
                        new_component = merged_component
                        idx = 0

                    if not proximity:
                        idx += 1


                self.components.append(new_component)

        filtered_components = []
        for component in self.components:

            area_check = 6000 < component.bbox['w'] * component.bbox['h'] < 300000
            width_check = 100 < component.bbox['w'] < 1200
            height_check = 100 < component.bbox['h'] < 1200

            if all((area_check, width_check, height_check)):
                filtered_components.append(component)

        self.components = filtered_components

    def output(self, original_image):
        '''Docstring'''

        if not self.components:
            return "No signatures were found on this document"
        else:
            output = original_image.copy()
            for component in self.components:
                x = component.bbox['x']
                y = component.bbox['y']
                xw = component.bbox['xw']
                yh = component.bbox['yh']
                cv2.rectangle(output, (x, y), (xw, yh), (0, 255, 0), 3)
            cv2.imshow("output", output)
            cv2.waitKey(0)
