import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, LayoutLMv3Processor
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import json
from PIL import Image

class DocumentDatasetUpdated(Dataset):
    def __init__(
        self,
        data: List[Dict],
        processor,
        max_length: int = 512,
        doc_type_map: Dict[str, int] = None
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.doc_type_map = doc_type_map or {'invoice': 0, 'resume': 1, 'report': 2}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = item.get('image')
            text = item.get('text', '')
            doc_type = item.get('document_type', 'report')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif image is None:
                image = Image.new('RGB', (224, 224), color='white')
            
            words = text.split()[:self.max_length - 2]  
            
            if not words:
                words = ['empty']
            img_width, img_height = image.size
            boxes = []
            words_per_line = 10
            
            for i, word in enumerate(words):
                line = i // words_per_line
                col = i % words_per_line
                
                x1 = int((col / words_per_line) * img_width)
                y1 = int((line / 20) * img_height)  # Assume max 20 lines
                x2 = min(x1 + int(img_width / words_per_line), img_width)
                y2 = min(y1 + int(img_height / 20), img_height)
                x2 = max(x2, x1 + 10)
                y2 = max(y2, y1 + 10)
                
                boxes.append([x1, y1, x2, y2])
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Squeeze batch dimension
            for k, v in encoding.items():
                if isinstance(v, torch.Tensor):
                    encoding[k] = v.squeeze(0)
            
            # Add label
            label = self.doc_type_map.get(doc_type, 2)
            encoding['labels'] = torch.tensor(label, dtype=torch.long)
            
            # Create entity labels (for demo, we'll create pseudo-labels)
            # In production, you'd have actual entity annotations
            entity_labels = self._create_entity_labels(words, text, doc_type)
            encoding['entity_labels'] = torch.tensor(entity_labels, dtype=torch.long)
            
            return encoding
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a dummy encoding
            return self._get_dummy_encoding()
    
    def _create_entity_labels(self, words: List[str], full_text: str, doc_type: str) -> List[int]:
        """
        Create pseudo entity labels for training
        Entity types: 0=O, 1=PERSON, 2=ORG, 3=DATE, 4=MONEY, 5=EMAIL, 
                     6=PHONE, 7=LOCATION, 8=TITLE, 9=SKILL
        """
        labels = []
        
        for word in words:
            label = 0  # Default: O (outside)
            word_lower = word.lower()
            
            # Simple heuristics for entity detection
            if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', word):
                label = 3  # DATE
            elif re.match(r'[$₹€£]\d+', word) or re.match(r'\d+\.\d{2}', word):
                label = 4  # MONEY
            elif '@' in word:
                label = 5  # EMAIL
            elif re.match(r'\d{3}[-.]?\d{3}[-.]?\d{4}', word):
                label = 6  # PHONE
            elif word[0].isupper() and len(word) > 2:
                # Capitalized words might be names or orgs
                if doc_type == 'resume':
                    label = 1  # PERSON
                elif doc_type == 'invoice':
                    label = 2  # ORG
            
            labels.append(label)
        
        # Pad or truncate to max_length
        if len(labels) < self.max_length:
            labels.extend([-100] * (self.max_length - len(labels)))
        else:
            labels = labels[:self.max_length]
        
        return labels
    
    def _get_dummy_encoding(self):
        """Return dummy encoding for error cases"""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'bbox': torch.zeros(self.max_length, 4, dtype=torch.long),
            'pixel_values': torch.zeros(3, 224, 224),
            'labels': torch.tensor(0, dtype=torch.long),
            'entity_labels': torch.full((self.max_length,), -100, dtype=torch.long)
        }

# ============================================================================
# DATA PREPARATION (Updated)
# ============================================================================

def prepare_training_data_updated(datasets_dict, processor, test_size=0.2, max_samples_per_type=100):
    """
    Prepare training and validation datasets from your specific datasets
    
    Args:
        datasets_dict: Dictionary with 'invoice', 'resume', 'report' keys
        processor: LayoutLMv3 processor
        test_size: Validation split ratio
        max_samples_per_type: Maximum samples per document type
    """
    print("\n" + "="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)
    
    all_data = []
    
    # Process each dataset type
    for doc_type, dataset in datasets_dict.items():
        if dataset is None or len(dataset) == 0:
            print(f"⚠ No data for {doc_type}")
            continue
        
        print(f"\nProcessing {doc_type} dataset...")
        
        # Limit samples
        samples = dataset[:max_samples_per_type]
        
        for item in samples:
            try:
                # Ensure required fields
                if 'image' not in item or 'text' not in item:
                    continue
                
                # Add document type if not present
                if 'document_type' not in item:
                    item['document_type'] = doc_type
                
                all_data.append(item)
                
            except Exception as e:
                print(f"  Error processing item: {e}")
                continue
        
        print(f"  ✓ Added {len([d for d in all_data if d['document_type'] == doc_type])} samples")
    
    if len(all_data) == 0:
        raise ValueError("No valid data found!")
    
    print(f"\n✓ Total samples: {len(all_data)}")
    
    # Print distribution
    from collections import Counter
    type_dist = Counter([d['document_type'] for d in all_data])
    print(f"Distribution: {dict(type_dist)}")
    
    # Split into train and validation
    try:
        train_data, val_data = train_test_split(
            all_data,
            test_size=test_size,
            random_state=42,
            stratify=[d['document_type'] for d in all_data]
        )
    except ValueError as e:
        print(f"⚠ Stratified split failed: {e}")
        print("Using random split...")
        train_data, val_data = train_test_split(
            all_data,
            test_size=test_size,
            random_state=42
        )
    
    print(f"\n✓ Train samples: {len(train_data)}")
    print(f"✓ Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = DocumentDatasetUpdated(train_data, processor)
    val_dataset = DocumentDatasetUpdated(val_data, processor)
    
    return train_dataset, val_dataset

# ============================================================================
# TRAINER CLASS (Same as before, compatible with updated dataset)
# ============================================================================

class DocumentAITrainer:
    """
    Complete training pipeline for Document AI model
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=2e-5,
        num_epochs=10,
        warmup_steps=500,
        save_dir='./checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    bbox=batch['bbox'],
                    pixel_values=batch.get('pixel_values'),
                    labels=batch.get('labels'),
                    entity_labels=batch.get('entity_labels')
                )
                
                loss = outputs['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Track metrics
                total_loss += loss.item()
                
                # Get predictions
                with torch.no_grad():
                    preds = torch.argmax(outputs['doc_logits'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    if 'labels' in batch:
                        all_labels.extend(batch['labels'].cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(len(self.train_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        bbox=batch['bbox'],
                        pixel_values=batch.get('pixel_values'),
                        labels=batch.get('labels'),
                        entity_labels=batch.get('entity_labels')
                    )
                    
                    loss = outputs['loss']
                    total_loss += loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(outputs['doc_logits'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    if 'labels' in batch:
                        all_labels.extend(batch['labels'].cpu().numpy())
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self):
        """Full training loop"""
        print("="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*80)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 80)
            
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(
                self.scheduler.get_last_lr()[0]
            )
            
            # Print metrics
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {self.history['learning_rate'][-1]:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss, val_acc)
                print(f"  ✓ Best model saved!")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss, val_acc)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        
        # Generate classification report
        if val_labels and len(set(val_labels)) > 1:
            print("\nFinal Validation Classification Report:")
            try:
                print(classification_report(
                    val_labels, 
                    val_preds,
                    target_names=['Invoice', 'Resume', 'Report'],
                    zero_division=0
                ))
            except Exception as e:
                print(f"Could not generate report: {e}")
        
        return self.history
    
    def save_checkpoint(self, filename, epoch, val_loss, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        import os
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        import os
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']
    
    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        if len(self.history['val_loss']) > 0 and len(self.history['train_loss']) > 0:
            loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
            axes[1, 1].plot(loss_diff)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Loss - Train Loss')
            axes[1, 1].set_title('Overfitting Monitor')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import re
    
    print("="*80)
    print("TRAINING PIPELINE - UPDATED FOR YOUR DATASETS")
    print("="*80)
    
    print("\nThis pipeline is compatible with:")
    print("  ✓ SROIE Invoice Dataset (Kaggle)")
    print("  ✓ datasetmaster/resumes")
    print("  ✓ CShorten/ML-ArXiv-Papers")
    
    print("\nTo use:")
    print("""
# 1. Load your datasets
from part_01_setup_updated import DocumentDatasetLoader, TextPreprocessor

loader = DocumentDatasetLoader()
datasets = loader.prepare_mixed_dataset()

# 2. Prepare data
from transformers import LayoutLMv3Processor
processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base')

train_dataset, val_dataset = prepare_training_data_updated(
    datasets, 
    processor,
    max_samples_per_type=100
)

# 3. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 4. Train
from part_02_model import MultiModalDocumentAI

model = MultiModalDocumentAI()
trainer = DocumentAITrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5
)

history = trainer.train()
trainer.plot_history()
    """)