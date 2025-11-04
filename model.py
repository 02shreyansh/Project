from transformers import BertTokenizer
from transformers import BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
all_dataset_items = []

for item in tqdm(all_data, desc="Tokenizing documents"):
    text = item['text']
    label = item['label']
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    all_dataset_items.append({
        'input_ids': encoding['input_ids'].squeeze(),
        'attention_mask': encoding['attention_mask'].squeeze(),
        'label': torch.tensor(label),
        'document_type': item['document_type'],
        'entities': item['entities'],
        'words': item['words'][:50],
        'boxes': item['boxes'][:50]
    })
train_size = int(0.8 * len(all_dataset_items))
val_size = len(all_dataset_items) - train_size

train_dataset_items = all_dataset_items[:train_size]
val_dataset_items = all_dataset_items[train_size:]

print(f"Training samples: {len(train_dataset_items)}")
print(f"Validation samples: {len(val_dataset_items)}")

bert_model = BertModel.from_pretrained('bert-base-uncased')
classifier = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 3) 
)

entity_extractor = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 128)
)
bert_model = bert_model.to(device)
classifier = classifier.to(device)
entity_extractor = entity_extractor.to(device)
print("Multi-modal transformer model built successfully!")

learning_rate = 2e-5
num_epochs = 3
batch_size = 8
optimizer = torch.optim.AdamW(
    list(bert_model.parameters()) +
    list(classifier.parameters()) +
    list(entity_extractor.parameters()),
    lr=learning_rate
)
criterion = nn.CrossEntropyLoss()
print("Training setup completed!")

def train_epoch(train_items, epoch):
    bert_model.train()
    classifier.train()
    entity_extractor.train()
    total_loss = 0
    correct = 0
    total = 0
    import random
    random.shuffle(train_items)
    for i in tqdm(range(0, len(train_items), batch_size), desc=f"Epoch {epoch+1}"):
        batch_items = train_items[i:i+batch_size]
        input_ids = torch.stack([item['input_ids'] for item in batch_items]).to(device)
        attention_mask = torch.stack([item['attention_mask'] for item in batch_items]).to(device)
        labels = torch.stack([item['label'] for item in batch_items]).to(device)
        optimizer.zero_grad()
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = classifier(pooled_output)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / (len(train_items) / batch_size)
    accuracy = correct / total

    return avg_loss, accuracy

def validate(val_items):
    bert_model.eval()
    classifier.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(val_items), batch_size):
            batch_items = val_items[i:i+batch_size]

            input_ids = torch.stack([item['input_ids'] for item in batch_items]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch_items]).to(device)
            labels = torch.stack([item['label'] for item in batch_items]).to(device)

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output

            logits = classifier(pooled_output)
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / (len(val_items) / batch_size)
    accuracy = correct / total

    return avg_loss, accuracy
print("Starting training...")
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(train_dataset_items, epoch)
    val_loss, val_acc = validate(val_dataset_items)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
print("\nTraining completed!")