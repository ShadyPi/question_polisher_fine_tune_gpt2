import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer

# Step 1: Prepare the data
train_data = [("What is the capital of France?", "Paris"), ("Who wrote Harry Potter?", "J.K. Rowling")]
validation_data = [("What is the largest planet in our solar system?", "Jupiter"),
                   ("Who painted the Mona Lisa?", "Leonardo da Vinci")]
test_data = [("What is the square root of 16?", "4"), ("What is the chemical symbol for gold?", "Au")]

# Step 3: Load the pre-trained GPT-2 model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)


# Step 5: Prepare the data for fine-tuning
class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question, answer = self.data[index]
        encoded = tokenizer.encode_plus(question, answer, add_special_tokens=True, max_length=512,
                                        pad_to_max_length=True, return_attention_mask=True, truncation=True)
        input_ids = torch.tensor(encoded['input_ids'])
        attention_mask = torch.tensor(encoded['attention_mask'])
        label = torch.tensor(tokenizer.encode(answer)[1:-1])  # Exclude the start-of-sequence and end-of-sequence tokens
        return input_ids, attention_mask, label


train_dataset = QADataset(train_data)
validation_dataset = QADataset(validation_data)


# Step 6: Define the fine-tuning model
class QuestionAnsweringModel(torch.nn.Module):
    def __init__(self, gpt2_model):
        super(QuestionAnsweringModel, self).__init__()
        self.gpt2 = gpt2_model
        self.classification_layer = torch.nn.Linear(gpt2_model.config.hidden_size,
                                                    2)  # Adjust the number of units based on the number of answer options

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.classification_layer(outputs[:, 0, :])  # Use the first token's representation for classification
        return logits


model = QuestionAnsweringModel(model)

# Step 7: Fine-tune the model
epochs = 5
batch_size = 4
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = []
    with torch.no_grad():
        for val_batch in validation_dataloader:
            val_input_ids, val_attention_mask, val_labels = val_batch
            val_input_ids = val_input_ids.to(device)
            val_attention_mask = val_attention_mask.to(device)
            val_labels = val_labels.to(device)
            val_logits = model(val_input_ids, val_attention_mask)
            val_loss.extend(loss_fn(val_logits, val_labels).item())

    print(f"Validation Loss: {torch.mean(torch.tensor(val_loss)):.4f}")

# Step 9: Test the model
test_dataset = QADataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
model.eval()
test_loss = []
with torch.no_grad():
    for test_batch in test_dataloader:
        test_input_ids, test_attention_mask, test_labels = test_batch
        test_input_ids = test_input_ids.to(device)
        test_attention_mask = test_attention_mask.to(device)
        test_labels = test_labels.to(device)
        test_logits = model(test_input_ids, test_attention_mask)
        test_loss.extend(loss_fn(test_logits, test_labels).item())

print(f"Test Loss: {torch.mean(torch.tensor(test_loss)):.4f}")

# Step 10: Deploy and use the model
# You can save the fine-tuned model using torch.save() and use it for inference on new question-answering tasks.
