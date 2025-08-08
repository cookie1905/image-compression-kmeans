import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ImageCompression:
    def __init__(self):
        pass
    
    def load_image(self, path):
        """
        Reads an image from the specified path using OpenCV.

        Parameters:
            path (str): The path to the image file.

        Returns:
            image (numpy.ndarray): The loaded image in BGR format, or None if loading fails.
        """
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def compress(self, path="images\\umbrella.jpg", k=5):
        image = self.load_image(path)
        pixel_values = image.reshape((-1, 3)) 
        pixel_values = np.float32(pixel_values)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(pixel_values)

        centers = np.uint8(kmeans.cluster_centers_)
        labels = kmeans.labels_
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title(f"KMeans Segmented (k={k})")
        plt.imshow(segmented_image)
        plt.axis("off")
        plt.savefig(f"images/result_{k}.png")
        plt.show()
        

image1 = ImageCompression()
image = image1.compress()
