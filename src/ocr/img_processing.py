import cv2
import numpy as np
from PIL import Image
import os

class ImagePreprocessor:
    """
    This class handles the preprocessing of images for OCR.
    It includes steps such as resizing, grayscale conversion, 
    thresholding, noise removal, and cropping the nutritional table.
    """

    def __init__(self, input_dir="data/img_raw", output_dir="data/img_processed"):
        """
        Initializes the ImagePreprocessor class with directories for input and output images.

        :param input_dir: Directory containing raw images.
        :param output_dir: Directory where processed images will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_image(self, file_path):
        """
        Load an image from the file path.
        
        :param file_path: Path to the image.
        :return: Image in OpenCV format.
        """
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Image {file_path} not found.")
        return img

    def save_image(self, img, file_name):
        """
        Save the processed image to the output directory.
        
        :param img: Processed image to be saved.
        :param file_name: File name to save as.
        """
        save_path = os.path.join(self.output_dir, file_name)
        cv2.imwrite(save_path, img)
        print(f"Image saved at {save_path}")
    
    def preprocess_image(self, img):
        """
        Preprocesses the input image to enhance OCR results.
        Steps include:
            1. Converting to grayscale.
            2. Denoising to remove artifacts.
            3. Thresholding to enhance contrast.
        
        :param img: The raw image.
        :return: Preprocessed image.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Denoising (using GaussianBlur to remove noise)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Thresholding (Binary Inverse Thresholding)
        # We apply binary threshold to get a black and white effect
        _, thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        
        return thresholded

    def crop_nutritional_table(self, img):
        """
        Crop the image to focus only on the nutritional table.
        This is a placeholder for a method that would locate the table and crop it.

        Approach: 
            1. Detect the largest rectangular contour (assuming it's the nutritional table).
            2. Crop the image based on that contour.

        :param img: Preprocessed image.
        :return: Cropped image focused on the nutritional table.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding rectangle
        cropped_img = img[y:y+h, x:x+w]
        
        return cropped_img

    def process_images(self):
        """
        This method processes all images from the input directory,
        applies preprocessing, and crops the nutritional table.
        This step does not involve applying any machine learning. However, ideally, it would be beneficial to incorporate ML to improve the detection of the nutritional table contours.
        """
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)
            
            try:
                # Step 1: Load the image
                img = self.load_image(file_path)
                
                # Step 2: Preprocess the image (grayscale, thresholding, etc.)
                preprocessed_img = self.preprocess_image(img)
                
                # Step 3: Crop the nutritional table
                cropped_img = self.crop_nutritional_table(preprocessed_img)

                # Here, we could also rotate the table to ensure that it has the right orientation
                
                # Step 4: Save the processed image
                self.save_image(cropped_img, file_name)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")


if __name__ == "__main__":
    # Instantiate the preprocessor
    preprocessor = ImagePreprocessor()
    
    # Process all images
    preprocessor.process_images()
