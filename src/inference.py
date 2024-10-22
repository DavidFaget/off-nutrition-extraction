import torch
from transformers import BertTokenizer
from model import MultitaskModel

def infer(model, tokenizer, text, device):
    """
    Performs inference on a single piece of text.

    :param model: The trained multitask model.
    :param tokenizer: BERT tokenizer.
    :param text: Input text to run inference on.
    :param device: Device to run the model on (CPU or GPU).
    :return: The predictions for each task.
    """
    model.eval()  # Set model to evaluation mode

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract predictions
    entity_predictions = torch.argmax(outputs['entity_logits'], dim=-1)
    single_label_prediction = torch.argmax(outputs['single_label_logits'], dim=-1)
    multi_label_predictions = (outputs['multi_label_logits'] > 0.5).int()

    return entity_predictions, single_label_prediction, multi_label_predictions

if __name__ == "__main__":
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskModel(pretrained_model_name="bert-base-uncased", num_entity_tags=3, num_single_labels=5, num_multi_labels=5)
    model.load_state_dict(torch.load('models/path_to_trained_model.pth', map_location=device))
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Input text for inference
    text = "Example text to run inference on."

    # Run inference
    entity_preds, single_label_pred, multi_label_preds = infer(model, tokenizer, text, device=device)

    # Print predictions
    print(f"Entity Predictions: {entity_preds}")
    print(f"Single-label Prediction: {single_label_pred}")
    print(f"Multi-label Predictions: {multi_label_preds}")
