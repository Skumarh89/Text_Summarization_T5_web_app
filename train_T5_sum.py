import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch.nn.functional as F


# Setting up the device for GPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataset
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = self.data[index]['article']
        highlights = self.data[index]['highlights']

        source = self.tokenizer.batch_encode_plus([article], max_length=self.source_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([highlights], max_length=self.summ_len, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for batch_idx, data in enumerate(loader):
        y = data['target_ids'].to(device, dtype=torch.long)
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y)

        # Calculate the loss
        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = y[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Print loss if desired
        if batch_idx % 500 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()






def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if batch_idx % 100 == 0:
                print(f'Completed batch {batch_idx}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def main():
    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 2
    TRAIN_EPOCHS = 5
    VAL_EPOCHS = 1
    LEARNING_RATE = 1e-4
    MAX_LEN = 512
    SUMMARY_LEN = 150

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_size = 0.8
    train_dataset = dataset["train"].select(range(int(len(dataset["train"]) * train_size)))
    val_dataset = dataset["train"].select(range(int(len(dataset["train"]) * train_size), len(dataset["train"])))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    print('Now generating summaries on our fine-tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv('predictions.csv')
        print('Output Files generated for review')
    
    # Save the trained model
    model.save_pretrained("saved_model")

    # Load the saved model
    loaded_model = T5ForConditionalGeneration.from_pretrained("saved_model")
    loaded_model = loaded_model.to(device)

    
if __name__ == '__main__':
    main()
