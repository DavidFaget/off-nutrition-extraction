import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MultitaskModel
from dataloader import MultitaskDataset, create_dataloader

def train(model, train_loader, optimizer, device, num_epochs=10):
    """
    Trains the multitask model.
    
    :param model: The multitask model to be trained.
    :param train_loader: DataLoader for the training set.
    :param optimizer: Optimizer for gradient descent.
    :param device: Device to run the model on (CPU or GPU).
    :param num_epochs: Number of epochs to train.
    """
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_tags = batch['entity_tags'].to(device)
            single_label = batch['single_label'].to(device)
            multi_label = batch['multi_label'].to(device)
            
            optimizer.zero_grad()  # Reset gradients
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute multitask loss
            loss = model.multitask_loss(predictions=outputs, entity_tags=entity_tags, single_label=single_label, multi_label=multi_label)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 10

    # Load data
    train_loader = create_dataloader(batch_size=batch_size, data_dir="data/splits/train", shuffle=True)

    # Initialize model, optimizer, and loss function
    model = MultitaskModel(pretrained_model_name="bert-base-uncased", num_entity_tags=3, num_single_labels=5, num_multi_labels=5)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, optimizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=num_epochs)
