import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration for BERT from scratch
config = BertConfig(
    vocab_size=30522,        # Standard BERT vocabulary size
    hidden_size=768,         # Embedding dimension
    num_hidden_layers=12,    # Number of transformer blocks
    num_attention_heads=12,  # Number of attention heads per block
    intermediate_size=3072,  # Feedforward layer dimension
    num_labels=2             # For binary classification (positive/negative)
)

# Initialize an untrained BERT model and tokenizer
model = BertForSequenceClassification(config).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the IMDb dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

# Tokenize and prepare datasets
encoded_train_dataset = train_dataset.map(preprocess_function, batched=True).remove_columns(["text"])
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True).remove_columns(["text"])
encoded_train_dataset.set_format("torch")
encoded_test_dataset.set_format("torch")

# Create data loaders
batch_size = 16
train_dataloader = DataLoader(encoded_train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(encoded_test_dataset, batch_size=batch_size)

# Optimizer and Loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 1
loss_values = []

# Check if training on GPU
if torch.cuda.is_available():
    print("Training on GPU")
else:
    print("Training on CPU")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    total_loss = 0
    
    model.train()  # Set model to training mode
    for batch in train_dataloader:
        # Move batch data to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}

        # Clear previously calculated gradients
        optimizer.zero_grad()

        # Perform a forward pass and compute the loss
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        loss = outputs.loss

        # Backpropagate to compute gradients
        loss.backward()

        # Update parameters with optimizer
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item()
    
    # Store average loss for this epoch
    avg_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_loss)
    print(f"Average loss for epoch {epoch + 1}: {avg_loss}")

print("Training completed.")

# Save the model and tokenizer
output_dir = "./bert_sentiment_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

