import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model and tokenizer
model_dir = "./bert_sentiment_model"
model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Load the IMDb dataset for testing
dataset = load_dataset("imdb")
test_dataset = dataset["test"]

# Preprocess the dataset
def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)
# Tokenize and prepare the test dataset
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True).remove_columns(["text"])
encoded_test_dataset.set_format("torch")
# Create a DataLoader for the test dataset
batch_size = 16
test_dataloader = DataLoader(encoded_test_dataset, batch_size=batch_size)
# Evaluation function to compute accuracy
def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader:
            # Move batch data to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            # Perform a forward pass and get logits
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            # Get the predicted class (the index with the highest logit score)
            predictions = torch.argmax(logits, dim=-1)
            # Count correct predictions by comparing with true labels
            correct_predictions += (predictions == batch['label']).sum().item()
            total_predictions += batch['label'].size(0)
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy
# Run evaluation
accuracy = evaluate(model, test_dataloader)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
