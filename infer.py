import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
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

# Load the test data
def load_test_data(filename):
    data = pd.read_csv(filename, delimiter='\t')
    return data

test_data = load_test_data('data_test.txt')

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset and dataloader
test_dataset = TextDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model_path = 'checkpoints/checkpoint_10000.pt'
model = torch.load(model_path)
model.eval()

# Evaluation function
def evaluate(model, loader):
    model.eval()
    total_mae = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, labels in loader:
            preds = model(input_ids, input_ids, input_ids)[2]
            preds = preds.squeeze().cpu().numpy()

            # Compute MAE and MSE
            mae = mean_absolute_error(preds, labels.cpu().numpy())
            mse = mean_squared_error(preds, labels.cpu().numpy())

            total_mae += mae * len(labels)
            total_mse += mse * len(labels)
            total_samples += len(labels)

    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    return avg_mae, avg_mse

# Perform evaluation
test_mae, test_mse = evaluate(model, test_loader)
print(f'Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}')
