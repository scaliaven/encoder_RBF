from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# Load dataset
dataset = load_dataset("/home/hh3043/Research/Durf_2023/yelp_review_full")
print(dataset["train"][100])

# Load tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert_model = AutoModel.from_pretrained("bert-base-cased")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Define a custom classifier on top of BERT encoder
class CustomClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(CustomClassifier, self).__init__()
        self.bert = bert_model
        self.additional_layer = nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, y,attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state 
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(outputs.shape[0], -1) #reshape the feature matrix of each video to a feature vector so that it can be put into the kernel
        y = np.array(y.cpu()) # y is an array consists of the "popularity" of videos
        kernel = 50.0**2 * RBF(length_scale=50.0)
        output = KernelRidge(alpha=1.0, kernel=kernel)
        output.fit(outputs, y)
        prediction = output.predict(outputs)
        return torch.from_numpy(prediction).float().to(device) 

# Create an instance of the custom classifier
num_labels = 5  # Modify based on your specific task
custom_model = CustomClassifier(bert_model, num_labels)

# Move the model to the desired device
custom_model.to(device)
# Create optimizer and scheduler
optimizer = AdamW(custom_model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(small_train_dataset) // 64  # Adjust batch size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

# Training loop
progress_bar = tqdm(range(num_training_steps))
custom_model.train()

for epoch in range(num_epochs):
    for batch in DataLoader(small_train_dataset, shuffle=True, batch_size=64):
        batch = {k: v.to(device) for k, v in batch.items()}
      
        outputs = custom_model(batch["input_ids"], batch["labels"])
        outputs.requires_grad = True
        loss = nn.MSELoss()(outputs, batch["labels"].float())      
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    #lr_scheduler.step()
    my_lr = lr_scheduler.get_last_lr()[0]
    print("loss:", loss, "lr", my_lr)

# Evaluation
metric = evaluate.load("accuracy")
custom_model.eval()

total_loss = 0.0
total_batches = 0

# Iterate through the test data
with torch.no_grad():
    for batch in DataLoader(small_eval_dataset, batch_size=64):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass
        outputs = custom_model(batch["input_ids"], batch["labels"])
        # Calculate the loss
        loss = nn.MSELoss()(outputs, batch["labels"].float())
        # Accumulate the loss and update the number of batches
        total_loss += loss.item()
        total_batches += 1

# Calculate the average loss over all batches
average_loss = total_loss / total_batches
# Print or use the average loss as needed
print("Average Test Loss:", average_loss)
