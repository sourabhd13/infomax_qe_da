import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import MMIM
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sentences = [row['S1'], row['S2'], row['S3']]
        labels = row['Normalized_score']

        # Tokenize and encode the sentences
        encoded_sentences = [self.tokenizer.encode(sent, add_special_tokens=True, truncation=True, padding='max_length', max_length=128) for sent in sentences]

        # Convert to tensors
        input_ids = torch.tensor([enc['input_ids'] for enc in encoded_sentences])
        token_type_ids = torch.tensor([enc['token_type_ids'] for enc in encoded_sentences])
        attention_mask = torch.tensor([enc['attention_mask'] for enc in encoded_sentences])

        return input_ids, token_type_ids, attention_mask, torch.tensor(labels)

# Load and preprocess the data
def load_data(filename):
    data = pd.read_excel(filename)
    return data

# Load the data
train_data = load_data('data_train.txt')
dev_data = load_data('data_dev.txt')
test_data = load_data('data_test.txt')

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets and dataloaders
train_dataset = TextDataset(train_data, tokenizer)
dev_dataset = TextDataset(dev_data, tokenizer)
test_dataset = TextDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Hyperparameters
hp = {
    'd_tin': 768,
    'd_tout': 768,
    'd_prjh': 512,
    'n_layer': 3,
    'dropout_prj': 0.2,
    'bidirectional': True,
    'mmilb_mid_activation': 'relu',
    'mmilb_last_activation': 'sigmoid',
    'cpc_layers': 3,
    'cpc_activation': 'relu',
    'n_class': 1
}

# BERT Config for bert_config1, bert_config2, bert_config3
bert_config = {
    'vocab_size': 30522,
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1
}


model = MMIM(hp, bert_config, bert_config, bert_config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for input_ids, token_type_ids, attention_mask, labels in loader:
        optimizer.zero_grad()

        lld, nce, preds, _, _ = model(input_ids, input_ids, input_ids, labels)
        loss = criterion(preds.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_mae = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, labels in loader:
            lld, nce, preds, _, _ = model(input_ids, input_ids, input_ids, labels)

            # Compute MAE and MSE
            mae = mean_absolute_error(preds.squeeze().cpu().numpy(), labels.cpu().numpy())
            mse = mean_squared_error(preds.squeeze().cpu().numpy(), labels.cpu().numpy())

            total_mae += mae * len(labels)
            total_mse += mse * len(labels)
            total_samples += len(labels)

    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    return avg_mae, avg_mse

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    dev_mae, dev_mse = evaluate(model, dev_loader, criterion)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Dev MAE: {dev_mae:.4f}, Dev MSE: {dev_mse:.4f}')

# Evaluation on test set
test_mae, test_mse = evaluate(model, test_loader, criterion)
print(f'Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}')

