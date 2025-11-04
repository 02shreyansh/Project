# End-to-End AI System for Intelligent Document Understanding

## 🎯 Project Overview

This is a comprehensive multi-component AI system that automatically extracts, understands, and makes decisions based on unstructured business documents (invoices, resumes, and reports).

### Key Features

✅ **Multi-Modal Learning**: Combines visual (layout, images) and textual features  
✅ **Document Classification**: Automatically identifies document types  
✅ **Information Extraction**: Extracts key fields and entities  
✅ **Intelligent Reasoning**: Validates data, ranks candidates, detects anomalies  
✅ **Explainable AI**: Provides attention maps, saliency visualizations, and feature importance  
✅ **REST API**: FastAPI deployment for production use  

---

## 🏗️ System Architecture

```
Input Document (Image/PDF)
         ↓
    OCR Engine (EasyOCR)
         ↓
Multi-Modal Transformer (LayoutLMv3)
    ↓           ↓           ↓
Classification  Entity    Reasoning
               Extraction   Layer
         ↓           ↓           ↓
    Decision Engine + Explainability
         ↓
    JSON Output + Visualizations
```

### Components

1. **Data Handling Layer** (`data_handling.py`)
   - OCR with EasyOCR
   - Text preprocessing with spaCy
   - Dataset loading from Hugging Face

2. **Multi-Modal Model** (`model.py`)
   - LayoutLMv3/Bert-based transformer
   - Visual + textual feature fusion
   - Classification + entity extraction + reasoning heads

3. **Explainability Module** (`Explain.py`)
   - Attention visualization
   - Feature importance analysis

5. **AI Reasoning Layer** (`Reason.py`)
   - Invoice validation
   - Resume ranking
   - Anomaly detection

6. **API Deployment** (`fastapi.py`)
   - REST API with FastAPI
   - Batch processing support
   - Health monitoring
---

## 📊 Sample Output

```json
{
  "document_id": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "document_type": "invoice",
  "fields_extracted": {
    "invoice_no": "INV-2025-321",
    "total_amount": "$58,400",
    "vendor": "ABC Solutions Pvt Ltd",
    "date": "01/15/2025",
    "subtotal": "$50,000",
    "tax": "$8,400"
  },
  "decision": "Valid",
  "confidence_score": 0.94,
  "reasoning": [
    "All required fields present",
    "Mathematical validation passed: subtotal + tax = total",
    "Date format valid"
  ],
  "validation_result": {
    "is_valid": true,
    "errors": [],
    "warnings": [],
    "confidence": 1.0
  },
  "anomalies": {
    "detected_anomalies": [],
    "overall_risk": "low"
  },
  "explainability_map": "./explanations/.../document_attention.png",
  "timestamp": "2025-11-03T10:30:45.123456"
}
```

---

## 🔧 Technical Details

### Model Architecture
- **Backbone**: LayoutLMv3/BERT
- **Input**: RGB images (224x224) + text tokens + bounding boxes
- **Output Heads**:
  - Document Classification (3 classes)
  - Entity Extraction (10 entity types)
  - Reasoning Validation (valid/invalid)
  
### Training Details

- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler**: Linear warmup + decay
- **Batch Size**: 4
- **Max Sequence Length**: 512 tokens
- **Device**: CUDA (if available) or CPU
---

## 🎨 Explainability Visualizations

The system generates multiple explainability visualizations:

1. **Attention Heatmap** - Shows token-to-token attention patterns
2. **Document Attention Overlay** - Highlights important regions on the original document
3. **Feature Importance** - Ranks the most important words for prediction
4. **Saliency Map** - Shows which pixels influenced the decision

---

## 🎓 Key Components 

### 1. Document Classification
Uses the CLS token representation to classify documents into:
- Invoice
- Resume
- Report

### 2. Entity Extraction
Token-level classification to identify:
- Person names
- Organizations
- Dates
- Monetary values
- Locations
- Custom entities (invoice numbers, email addresses, etc.)

### 3. Reasoning Layer
Neural network that:
- Validates extracted information
- Checks mathematical consistency
- Detects anomalies
- Estimates confidence

### 4. Decision Engine
Rule-based + neural reasoning for:
- **Invoices**: Validates totals, dates, required fields
- **Resumes**: Ranks candidates based on job requirements
- **papers**: Checks completeness and structure

---