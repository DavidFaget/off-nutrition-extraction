import torch
from model import MultitaskModel
from dataloader import create_dataloader

def evaluate(model, eval_loader, device):
    """
    Evaluates the multitask model on the validation or test set.
    
    :param model: The multitask model.
    :param eval_loader: DataLoader for the evaluation set.
    :param device: Device to run the model on (CPU or GPU).
    :return: The evaluation loss.
    """
    model.eval()  # Set model to evaluation mode
    eval_loss = 0.0

    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_tags = batch['entity_tags'].to(device)
            single_label = batch['single_label'].to(device)
            multi_label = batch['multi_label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute multitask loss
            loss = model.multitask_loss(predictions=outputs, entity_tags=entity_tags, single_label=single_label, multi_label=multi_label)
            eval_loss += loss.item()

    return eval_loss / len(eval_loader)

if __name__ == "__main__":
    # Load evaluation data
    eval_loader = create_dataloader(batch_size=16, data_dir="data/splits/test", shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskModel(pretrained_model_name="bert-base-uncased", num_entity_tags=3, num_single_labels=5, num_multi_labels=5)
    model.load_state_dict(torch.load('models/path_to_trained_model.pth', map_location=device))
    model = model.to(device)

    # Evaluate the model
    eval_loss = evaluate(model, eval_loader, device=device)
    print(f"Evaluation Loss: {eval_loss}")
