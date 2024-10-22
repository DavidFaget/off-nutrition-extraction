import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer

class MultitaskDataset(Dataset):
    """
    This class represents a dataset designed for multitask learning:
    - Entity Tagging
    - Single-label Classification
    - Multi-label Classification
    """
    
    # IMPORTANT: CHANGE THESE DIRECTORIES IF WE WANT TO READ SPLITTED DATA
    def __init__(self, data_dir="data/ocr_processed", tokenizer_name="bert-base-uncased", max_len=256):
        """
        Initializes the dataset by loading and preparing the text data.

        :param data_dir: Directory containing processed text data.
        :param tokenizer_name: Tokenizer to use for converting text to tokens.
        :param max_len: Maximum length for token sequences.
        """
        self.data_dir = data_dir
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Load all text files in the directory
        self.samples = self._load_data()
        
        # Labels for single-label classification task
        self.single_labels = ['Low Fat', 'High Protein', 'Vegan', 'Keto', 'Other']
        
        # Labels for multi-label classification task (can overlap, e.g., a food might be both "Low Fat" and "Vegan")
        self.multi_labels = ['Low Sugar', 'Gluten Free', 'Organic', 'Non-GMO', 'Dairy Free']

        # Prepare a multi-label binarizer for converting multi-labels to binary vectors
        self.mlb = MultiLabelBinarizer(classes=self.multi_labels)

    def _load_data(self):
        """
        Loads all text files from the data directory.
        
        :return: A list of text samples from the files.
        """
        samples = []
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r') as file:
                text = file.read()
                samples.append(text)
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def _get_entity_tags(self, text):
        """
        Placeholder function for generating entity tags.
        In a real system, this function would generate entity tags based on the text.
        
        :param text: Input text.
        :return: A list of entity tags.
        """
        # TODO: Replace with actual entity tagging logic
        return [0] * len(text.split())  # Example: Return '0' for each token (no entity tagged)

    def _get_single_label(self, text):
        """
        Placeholder function for generating single-label classification.
        This simulates assigning one label to the entire text.

        :param text: Input text.
        :return: A single-label classification.
        """
        # TODO: Replace with actual single-label classification logic
        return torch.tensor(0)  # Example: Always returns label '0'

    def _get_multi_labels(self, text):
        """
        Placeholder function for generating multi-label classification.
        
        :param text: Input text.
        :return: A binary vector representing multiple labels.
        """
        # TODO: Replace with actual multi-label classification logic
        return self.mlb.transform([['Low Sugar', 'Organic']])[0]  # Example: Two labels

    def __getitem__(self, index):
        """
        Retrieves a data point from the dataset, tokenizes it, and prepares it for multitask learning.
        
        :param index: Index of the sample to retrieve.
        :return: A dictionary containing input tokens, entity tags, single label, and multi-label.
        """
        text = self.samples[index]

        # Tokenize the text
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Get entity tags (for entity tagging task)
        entity_tags = torch.tensor(self._get_entity_tags(text), dtype=torch.long)

        # Get single-label classification
        single_label = self._get_single_label(text)

        # Get multi-label classification
        multi_label = torch.tensor(self._get_multi_labels(text), dtype=torch.float32)

        return {
            'input_ids': tokens['input_ids'].squeeze(0),  # Tensor of input IDs
            'attention_mask': tokens['attention_mask'].squeeze(0),  # Tensor of attention masks
            'entity_tags': entity_tags,  # Tensor of entity tags
            'single_label': single_label,  # Tensor for single-label classification
            'multi_label': multi_label  # Tensor for multi-label classification
        }

def create_dataloader(batch_size=32, shuffle=True, data_dir="data/ocr_processed"):
    """
    Function to create and return a DataLoader for multitask learning.

    :param batch_size: The number of samples per batch.
    :param shuffle: Whether to shuffle the data after every epoch.
    :param data_dir: Directory containing the processed text data.
    :return: DataLoader instance for training the multitask model.
    """
    dataset = MultitaskDataset(data_dir=data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
