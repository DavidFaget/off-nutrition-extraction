# Nutritional Table OCR and Multitask Learning

## Project Overview and workflow

This project aims to extract and process nutritional information from images of nutritional tables using Optical Character Recognition (OCR). Then, it employs a multitask learning model that performs entity tagging, single-label classification, and multi-label classification. 

The project is structured into two main phases:
1- OCR: The input consists of raw images, while the output is the processed text.
2- Entity Tagging and Classification: The input is the processed text, and the output consists of model predictions.

### Step 1 Overview:

The OCR process is divided into three stages. First, we preprocess the raw images to prepare them for OCR. Next, we utilize pre-trained models to identify text regions and extract the text. Finally, we process the extracted text to make it suitable for entity tagging and classification.

For the text extraction, we use a combination of CTPN (connectionist text proposal network) to detect text regions and Tesseract to transcribe the text. The benefit of using CTPN instead of relying solely on Tesseract is that CTPN is designed to detect text regions in complex and disorganized images, such as tables with unclear or misaligned cells (which is our case). While Tesseract works well when the text is already neatly segmented, CTPN precisely identifies the areas containing text, improving segmentation in challenging situations. This allows Tesseract to be applied more effectively only to the relevant regions, reducing errors that could arise from trying to recognize text in irrelevant areas.

### Step 2 Overview:

The input for this second phase is the output from the first phase. We enable the splitting of textual data in various custom ways, which are then processed using a PyTorch dataloader. A multitask model is defined to manage entity tagging, single-label classification, and multi-label classification. Additionally, we provide scripts for training, evaluating, and making predictions with the model.

For the multitask model, we employ a BERT backbone and three different heads for each task. While more recent models like GPT-4 have advanced the state of the art, BERT remains the ideal choice for multitask models involving entity tagging and classification (both single and multi-label) due to its efficiency, robustness, and versatility. BERT's bi-directional Transformer architecture captures deep contextual representations, important for understanding entities and discerning relationships in classification tasks. Although adaptations like BioBERT and SciBERT excel in specific domains, they focus primarily on entity tagging, which limits their effectiveness in classification. By using BERT, I can leverage its multi-task learning capabilities, seamlessly integrating entity recognition and classification without switching models. Additionally, BERT's established fine-tuning process benefits from ample resources and pre-trained checkpoints. While newer models like GPT-4 are powerful, they often come with significant computational overhead for tasks that don’t require generative outputs. Thus, BERT strikes the right balance between performance, efficiency, and ease of deployment, making it highly suitable for my multitask learning scenario.


## Project Structure

```
off-nutrition-extraction/
│
├── data/
│   ├── img_raw/              # Directory for raw images of nutritional tables
│   ├── img_processed/        # Directory for processed images ready for OCR
│   ├── ocr_raw/              # Directory for raw text output from OCR
│   ├── ocr_processed/        # Directory for processed text ready for training
│   └── splits/               # Directory containing training, validation, and test splits
│       ├── train/            # Training dataset
│       ├── val/              # Validation dataset
│       └── test/             # Test dataset
│
├── src/                  # Directory containing all scripts
│   ├── ocr/                  # Directory for OCR-related scripts
│   │   ├── img_preprocessing.py   # Preprocess images for OCR
│   │   ├── ocr.py                 # Perform OCR on processed images
│   │   └── text_preprocessing.py   # Process OCR text for training. Also run img_preprocessing.py and ocr.py.
│   ├── dataloader.py          # Handle data loading for training and evaluation
│   ├── data_splitting.py       # Split data into training, validation, and test sets
│   ├── model.py               # Define the multitask learning model
│   ├── train.py               # Script for training the model
│   ├── evaluate.py            # Evaluate the model's performance
│   └── inference.py           # Perform inference on new data
│
├── models/                   # Directory for trained models
├── requirements.txt          # Contains the requirements to run the scripts
└── README.md                 # Project documentation

```

## Script Descriptions

### 1. `img_preprocessing.py`

- **Purpose**: This script preprocesses images of nutritional tables for OCR. It includes steps such as resizing, converting to grayscale, thresholding, and cropping to focus on the nutritional content.
- **Output**: Processed images are stored in `data/img_processed/`.

### 2. `ocr.py`

- **Purpose**: This script uses a combination of a text detection algorithm (CTPN) and Tesseract OCR to extract text from processed images.
- **Output**: Extracted raw text is saved to `data/ocr_raw/`.

### 3. `text_processing.py`

- **Purpose**: This script processes the raw text obtained from OCR. It cleans and formats the text for training the model, preparing it for entity tagging and classification.
- **Execution**: It automatically runs `img_preprocessing.py` and `ocr.py` as part of its execution flow (it runs the whole first step of the workflow).
- **Output**: Processed text is saved to `data/ocr_processed/`.

### 4. `dataloader.py`

- **Purpose**: This script defines the `MultitaskDataset` class, which loads and preprocesses the data for training and evaluation. It creates DataLoader instances for batching data.
- **Functions**:
  - `create_dataloader`: Function to create DataLoader instances for training and evaluation datasets.

### 5. `data_splitting.py`

- **Purpose**: This script handles the splitting of the dataset into training, validation, and test sets. Custom splitting methods can be implemented as needed.

### 6. `model.py`

- **Purpose**: This script defines the `MultitaskModel`, a neural network model for multitask learning.
- **Architecture**: The model consists of a shared BERT backbone with three task-specific heads for:
  - Entity tagging (token-level classification)
  - Single-label classification
  - Multi-label classification

### 7. `train.py`

- **Purpose**: This script trains the `MultitaskModel` on the training dataset. It manages the training loop, including forward passes, loss computation, and backpropagation.
- **Output**: Trained model weights are saved for future use.

### 8. `evaluate.py`

- **Purpose**: This script evaluates the trained model on a validation or test dataset. It computes and returns the evaluation loss.
- **Output**: Loss metrics are printed to the console for performance assessment.

### 9. `inference.py`

- **Purpose**: This script runs inference on new text inputs using the trained model. It outputs the predictions for entity tagging, single-label classification, and multi-label classification.
- **Usage**: Users can input new text to see predictions without retraining the model.

## How to Run the Project

1. **Setup Environment**: Make sure you have the necessary libraries installed. You may use a `requirements.txt` file or a `conda` environment.
2. **Text Extraction**: Run `python scripts/text_processing.py` to run all the first step (extract and process text from raw images). Note that we could also run img_processing.py and ocr.py independently.
3. **Train the Model**: Execute `python scripts/train.py` to train the multitask learning model on the processed data.
4. **Evaluate the Model**: Use `python scripts/evaluate.py` to assess model performance on validation or test data.
5. **Perform Inference**: Finally, run `python scripts/inference.py` to make predictions on new data (remember to extract and process the text before running inference).

## Future Work:

- Add config.yaml
- Include functions to test each part separately
- Train and develop a specific OCR model for our task
- Our current solution requires 2 steps. We could modify the inference script to automatically perform text extraction from raw images (by calling text_processing) and model inference.