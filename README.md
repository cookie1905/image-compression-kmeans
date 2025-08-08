# KMeans Image Compression

This project demonstrates a simple image compression technique using KMeans clustering.

## Description

The script loads an image and compresses it by reducing the number of colors using KMeans clustering. It segments the image colors into `k` clusters (default is 5), resulting in a simplified version of the original image with fewer colors.

## Features

- Load an image from a file path
- Compress the image by color quantization using KMeans clustering
- Display the original and compressed images side by side

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- Matplotlib

You can install the required packages using pip:

```bash
pip install opencv-python numpy scikit-learn matplotlib
