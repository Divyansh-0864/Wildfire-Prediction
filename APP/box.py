import cv2
import numpy as np
import os

def draw_fire_bbox(image_path):
    """
    Draws a bounding box around the fire region in the satellite image and saves the fire mask.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load the image from path: {image_path}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 50], dtype=np.uint8)
    upper_bound = np.array([180, 50, 255], dtype=np.uint8)

    fire_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    bbox_output_path = f"Imagefile/{base_name}-bbox.jpg"
    mask_output_path = f"Imagefile/{base_name}-mask.jpg"

    cv2.imwrite(bbox_output_path, image)
    cv2.imwrite(mask_output_path, fire_mask)

    return bbox_output_path, mask_output_path


# Example usage
bbox, mask = draw_fire_bbox('TestImg/-62.39074,51.35761.jpg')
bb = cv2.imread(bbox)
mask = cv2.imread(mask)

cv2.imshow("BBox", bb)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()