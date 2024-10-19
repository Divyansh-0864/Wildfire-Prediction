from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model('models/WildFireDetector.keras')

# Serve images from the Imagefile directory
@app.route('/Imagefile/<filename>')
def serve_image(filename):
    return send_from_directory('Imagefile', filename)

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

@app.route('/')
def html_file():
    return render_template('Main.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    image_upload = request.files['image']
    file_path = 'Imagefile/' + image_upload.filename

    if not os.path.exists('Imagefile/'):
        os.makedirs('Imagefile/')

    image_upload.save(file_path)
    process_img = Image.open(file_path)
    process_img = process_img.resize((350, 350))
    process_img = np.array(process_img) / 255.0
    process_img = np.expand_dims(process_img, axis=0)

    predictions = model.predict(process_img)

    if predictions[0][1] > 0.90:
        result = f'Wildfire Occurring! (Probability: {predictions[0][1]:.2f})'
        bbox_image, mask_image = draw_fire_bbox(file_path)

        # Send the correct URLs for the images
        bbox_image_url = f'/Imagefile/{os.path.basename(bbox_image)}'
        mask_image_url = f'/Imagefile/{os.path.basename(mask_image)}'

        return jsonify({'result': result, 'bbox_image': bbox_image_url, 'mask_image': mask_image_url})
    elif predictions[0][1] < 0.03:
        result = f'Looks like the place is safe (Probability: {predictions[0][1]:.2f})'
    else:
        result = f'Probability: {predictions[0][1]:.2f}'

    os.remove(file_path)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
