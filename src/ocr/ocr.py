import os
import cv2
import pytesseract
import numpy as np

class OCRProcessor:
    """
    This class handles the process of text detection (using CTPN) 
    and text extraction (using Tesseract) from processed images.
    """

    def __init__(self, input_dir="data/img_processed", output_dir="data/ocr_raw"):
        """
        Initializes the OCRProcessor class with directories for input images and output text files.

        :param input_dir: Directory containing processed images.
        :param output_dir: Directory where extracted text will be saved.
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
        :return: Loaded image.
        """
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Image {file_path} not found.")
        return img

    def save_text(self, text, file_name):
        """
        Save the extracted text to the output directory as a .txt file.

        :param text: Extracted text to save.
        :param file_name: The name of the output text file.
        """

        with open(self.output_dir, "w") as file:
            file.write(text)
        print(f"Text saved at {self.output_dir}")

    def detect_text_regions(self, img):
        """
        Detect text regions in the image using a text detection algorithm, such as CTPN.
        
        This function is a placeholder for the CTPN model.

        :param img: The input image.
        :return: List of bounding boxes where text is detected. Each box is represented as [x, y, w, h].
        """
        # Placeholder for CTPN text detection model (this part should be replaced with CTPN implementation).
        # For now, let's assume the function returns a list of bounding boxes around detected text.
        
        text_boxes = []
        
        # TODO: Implement CTPN model for detecting text regions. We can use a pre-trained model here.
        
        return text_boxes

    def extract_text_with_tesseract(self, img, text_boxes):
        """
        Extract text from the detected text regions using Tesseract OCR.

        :param img: The input image.
        :param text_boxes: List of bounding boxes around detected text regions.
        :return: Extracted text from all regions combined.
        """
        extracted_text = ""
        
        for box in text_boxes:
            x, y, w, h = box
            
            # Crop the text region from the image
            text_region = img[y:y+h, x:x+w]
            
            # Convert the region to grayscale
            gray_text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            
            # Perform OCR on the cropped text region using Tesseract
            # PSM 6 assumes a block of text. We can add languages here as an argument too (useful for our situation).
            text = pytesseract.image_to_string(gray_text_region, config='--psm 6')  
            
            # Append the text from the region to the final output
            extracted_text += text + "\n"
        
        return extracted_text

    def process_images(self):
        """
        Process all images in the input directory:
            1. Detect text regions using CTPN.
            2. Apply Tesseract to extract text from the detected regions.
            3. Save the extracted text to the output directory.
        """
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)
            
            try:
                # Step 1: Load the image
                img = self.load_image(file_path)
                
                # Step 2: Detect text regions (using CTPN)
                text_boxes = self.detect_text_regions(img)
                
                # Step 3: Extract text from the detected regions (using Tesseract)
                extracted_text = self.extract_text_with_tesseract(img, text_boxes)
                
                # Step 4: Save the extracted text
                self.save_text(extracted_text, file_name)
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")


if __name__ == "__main__":
    # Initialize the OCRProcessor
    ocr_processor = OCRProcessor()
    
    # Process all images for OCR
    ocr_processor.process_images()
