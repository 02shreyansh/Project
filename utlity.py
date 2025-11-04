save_dir = '/content/saved_models'
os.makedirs(save_dir, exist_ok=True)
torch.save(bert_model.state_dict(), os.path.join(save_dir, 'bert_model.pt'))
torch.save(classifier.state_dict(), os.path.join(save_dir, 'classifier.pt'))
torch.save(entity_extractor.state_dict(), os.path.join(save_dir, 'entity_extractor.pt'))
tokenizer.save_pretrained(save_dir)
with open(os.path.join(save_dir, 'label_map.json'), 'w') as f:
    json.dump({'label_map': label_map, 'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()}}, f)
print(f"Models saved to {save_dir}")


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/content/training_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training metrics visualization saved!")



from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate on full validation set
print("\nGenerating Evaluation Report...")

all_predictions = []
all_labels = []

bert_model.eval()
classifier.eval()

with torch.no_grad():
    for item in tqdm(val_dataset_items, desc="Evaluating"):
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = classifier(outputs.pooler_output)

        prediction = torch.argmax(logits, dim=1).item()
        all_predictions.append(prediction)
        all_labels.append(item['label'].item())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)

# Use labels parameter to explicitly define all classes (0, 1, 2)
target_names = ['invoice', 'resume', 'paper']
report = classification_report(all_labels, all_predictions,
                               labels=[0, 1, 2],  # ✅ Add this
                               target_names=target_names,
                               zero_division=0)  # Handle missing classes

print("\n" + "="*60)
print("FINAL EVALUATION REPORT")
print("="*60)
print(f"\nOverall Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Confusion Matrix with all classes
cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEvaluation completed!")

# Debug: Check class distribution
print("\nClass Distribution:")
print(f"Unique predicted classes: {set(all_predictions)}")
print(f"Unique actual classes: {set(all_labels)}")
print(f"Prediction counts: {[(i, all_predictions.count(i)) for i in range(3)]}")

summary = {
    "project_title": "End-to-End AI System for Intelligent Document Understanding",
    "completion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "datasets_used": {
        "invoices": len(invoice_data),
        "resumes": len(resume_data),
        "papers": len(papers_data),
        "total": len(all_data)
    },
    "model_architecture": {
        "base_model": "BERT (bert-base-uncased)",
        "classification_head": "3-layer MLP",
        "entity_extraction_head": "2-layer MLP",
        "total_parameters": sum(p.numel() for p in bert_model.parameters())
    },
    "training_configuration": {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW"
    },
    "performance_metrics": {
        "final_train_accuracy": round(train_accuracies[-1], 4),
        "final_val_accuracy": round(val_accuracies[-1], 4),
        "final_train_loss": round(train_losses[-1], 4),
        "final_val_loss": round(val_losses[-1], 4)
    },
    "features_implemented": [
        "Multi-modal document understanding (text + layout)",
        "OCR integration (EasyOCR)",
        "Document classification (invoice/resume/paper)",
        "Entity extraction",
        "AI reasoning layer",
        "Explainability (Attention heatmaps)",
        "REST API deployment (FastAPI)",
        "Decision-making capabilities"
    ],
    "api_endpoint": str("public_url") if 'public_url' in locals() else "Not started",
    "saved_models_path": save_dir
}

with open('/content/project_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
