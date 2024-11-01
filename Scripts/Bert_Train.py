import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig(
        vocab_size=30522,        # Standard BERT vocabulary size
        hidden_size=768,         # Embedding dimension
        num_hidden_layers=12,    # Number of transformer blocks
        num_attention_heads=12,  # Number of attention heads per block
        intermediate_size=3072,  # Feedforward layer dimension
        num_labels=2,            # For binary classification (positive/negative)
        hidden_dropout_prob=0.1, # Dropout for hidden layers
        attention_probs_dropout_prob=0.1 # Dropout for attention layers
        )

model = BertForSequenceClassification(config).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True).remove_columns(["text"])
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True).remove_columns(["text"])
encoded_train_dataset.set_format("torch")
encoded_test_dataset.set_format("torch")
batch_size = 16
train_dataloader = DataLoader(encoded_train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(encoded_test_dataset, batch_size=batch_size)
optimizer = AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
total_steps = len(train_dataloader) * 10  # Assume 10 epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 40
loss_values = []
if torch.cuda.is_available():
    print("Training on GPU")
else:
    print("Training on CPU")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    total_loss = 0
    model.train()  
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_loss)
    print(f"Average loss for epoch {epoch + 1}: {avg_loss}")
print("Training completed.")
output_dir = "./bert_sentiment_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")


flag = 5
if flag == 5:
    del model
    print('Model deleted from GPU')
    flag -= 1
