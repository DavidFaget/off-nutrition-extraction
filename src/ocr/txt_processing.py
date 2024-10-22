import os
import subprocess
import re

class TextProcessor:
    """
    This class handles the processing of raw OCR text. It cleans the text and prepares it
    for later tasks like entity tagging and classification. 
    """

    def __init__(self, input_dir="data/ocr_raw", output_dir="data/ocr_processed"):
        """
        Initializes the TextProcessor class with directories for input raw text and output processed text.

        :param input_dir: Directory containing raw OCR text.
        :param output_dir: Directory where processed text will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def clean_text(self, text):
        """
        Cleans the OCR-extracted text. 
        Typical cleaning steps might include:
            - Removing non-alphanumeric characters.
            - Normalizing whitespace.
            - Correcting OCR errors (common misread characters).
            - Removing header/footer information not related to the nutritional table.

        :param text: Raw text from OCR.
        :return: Cleaned and normalized text.
        """
        # Remove any non-printable characters
        text = re.sub(r'[^a-zA-Z0-9\s\.\,]', '', text)
        
        # Normalize multiple spaces or line breaks to single spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace common OCR misreads
        text = text.replace('O', '0')  # Example: Replace 'O' with '0' (if 'O' is often misread)
        text = text.replace('l', '1')  # Replace lowercase 'l' with '1' (OCR confusion)
        
        # Further domain-specific cleaning could be applied here based on the structure of nutritional data
        # For example, removing known table headers or common unwanted terms.
        return text

    def process_text_files(self):
        """
        Processes all the raw text files in the input directory, applies text cleaning, and saves 
        the cleaned text to the output directory.
        """
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)

            try:
                # Load the raw text
                with open(file_path, 'r') as file:
                    raw_text = file.read()

                # Clean the text
                cleaned_text = self.clean_text(raw_text)

                # Save the cleaned text
                self.save_processed_text(cleaned_text, file_name)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    def save_processed_text(self, text, file_name):
        """
        Save the processed text to the output directory.

        :param text: Processed text.
        :param file_name: The name of the output text file.
        """
        output_file_path = os.path.join(self.output_dir, file_name)

        with open(output_file_path, 'w') as file:
            file.write(text)
        print(f"Processed text saved at {output_file_path}")

    def run_preprocessing_pipeline(self):
        """
        Run the image preprocessing and OCR pipeline sequentially.
        Executes the `img_preprocessing.py` and `ocr.py` scripts.
        """
        try:
            # Step 1: Run img_preprocessing.py
            print("Running image preprocessing...")
            subprocess.run(['python', 'img_preprocessing.py'], check=True)
            
            # Step 2: Run ocr.py
            print("Running OCR...")
            subprocess.run(['python', 'ocr.py'], check=True)
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running preprocessing scripts: {str(e)}")

if __name__ == "__main__":
    # Step 1: Run the preprocessing pipeline
    text_processor = TextProcessor()
    text_processor.run_preprocessing_pipeline()

    # Step 2: Process the OCR-extracted raw text
    print("Processing OCR raw text...")
    text_processor.process_text_files()
