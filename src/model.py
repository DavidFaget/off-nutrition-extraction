import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MultitaskModel(nn.Module):
    """
    A multitask learning model for entity tagging, single-label classification, and multi-label classification.
    """
    
    def __init__(self, pretrained_model_name="bert-base-uncased", num_entity_tags=3, num_single_labels=5, num_multi_labels=5):
        """
        Initializes the multitask model with a shared BERT encoder and separate heads for each task.

        :param pretrained_model_name: Name of the pre-trained BERT model.
        :param num_entity_tags: Number of entity tags for token classification.
        :param num_single_labels: Number of classes for single-label classification.
        :param num_multi_labels: Number of binary labels for multi-label classification.
        """
        super(MultitaskModel, self).__init__()
        
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        
        # The following heads should be adjusted and we should try more configurations.
        # Entity tagging (token-level classification) head
        self.entity_tagging_head = nn.Linear(self.bert.config.hidden_size, num_entity_tags)
        
        # Single-label classification head
        self.single_label_classification_head = nn.Linear(self.bert.config.hidden_size, num_single_labels)
        
        # Multi-label classification head
        self.multi_label_classification_head = nn.Linear(self.bert.config.hidden_size, num_multi_labels)

        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(p=0.3)

        # Activation for multi-label classification (we keep softmax for single-label classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, entity_tags=None, single_label=None, multi_label=None):
        """
        Forward pass through the shared BERT encoder and the task-specific heads.

        :param input_ids: Token IDs for BERT input.
        :param attention_mask: Attention masks for input tokens.
        :param entity_tags: Ground truth entity tags (optional, for loss calculation).
        :param single_label: Ground truth single-label classification (optional, for loss calculation).
        :param multi_label: Ground truth multi-label classification (optional, for loss calculation).
        :return: A dictionary containing the predictions for each task.
        """
        # BERT output (last hidden states)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim] (pooled [CLS] token)

        # Entity tagging (token-level classification)
        entity_logits = self.entity_tagging_head(sequence_output)  # [batch_size, seq_len, num_entity_tags]

        # Single-label classification (using pooled [CLS] token)
        pooled_output_dropout = self.dropout(pooled_output)
        single_label_logits = self.single_label_classification_head(pooled_output_dropout)  # [batch_size, num_single_labels]

        # Multi-label classification (also using pooled [CLS] token)
        multi_label_logits = self.multi_label_classification_head(pooled_output_dropout)  # [batch_size, num_multi_labels]
        multi_label_logits = self.sigmoid(multi_label_logits)  # Apply sigmoid for multi-label classification

        # Output the predictions
        outputs = {
            'entity_logits': entity_logits,  # For entity tagging
            'single_label_logits': single_label_logits,  # For single-label classification
            'multi_label_logits': multi_label_logits  # For multi-label classification
        }

        return outputs

    def multitask_loss(self, predictions, entity_tags=None, single_label=None, multi_label=None):
        """
        Computes the multitask loss by combining the losses from each task.

        :param predictions: A dictionary of predicted logits.
        :param entity_tags: Ground truth entity tags for token classification.
        :param single_label: Ground truth single-label classification.
        :param multi_label: Ground truth multi-label classification.
        :return: Combined multitask loss.
        """
        # Cross-entropy loss for entity tagging (token-level classification)
        entity_loss = 0
        if entity_tags is not None:
            criterion_entity = nn.CrossEntropyLoss()
            entity_loss = criterion_entity(predictions['entity_logits'].view(-1, predictions['entity_logits'].size(-1)), entity_tags.view(-1))

        # Cross-entropy loss for single-label classification
        single_label_loss = 0
        if single_label is not None:
            criterion_single_label = nn.CrossEntropyLoss()
            single_label_loss = criterion_single_label(predictions['single_label_logits'], single_label)

        # Binary cross-entropy for multi-label classification
        multi_label_loss = 0
        if multi_label is not None:
            criterion_multi_label = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification
            multi_label_loss = criterion_multi_label(predictions['multi_label_logits'], multi_label)

        # Combine the losses (simple sum here, but we should optimize this)
        total_loss = entity_loss + single_label_loss + multi_label_loss

        return total_loss
