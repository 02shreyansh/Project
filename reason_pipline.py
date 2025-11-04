def predict_document(text, words, boxes, entities):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    ).to(device)
    bert_model.eval()
    classifier.eval()
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            output_attentions=True
        )

        pooled_output = outputs.pooler_output
        logits = classifier(pooled_output)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        document_type = reverse_label_map[predicted_class]
        attention_weights = outputs.attentions[-1][0].mean(dim=0)
        heatmap_path = generate_attention_heatmap(text, attention_weights)
        decision, reasoning_score = apply_reasoning(document_type, text, entities)
        output = {
            "document_type": document_type,
            "fields_extracted": entities,
            "decision": decision,
            "confidence_score": round(confidence * reasoning_score, 2),
            "explainability_map": heatmap_path,
            "reasoning_details": {
                "classification_confidence": round(confidence, 2),
                "reasoning_confidence": round(reasoning_score, 2)
            }
        }
        return output
print("Inference function defined!")

test_samples = val_dataset_items[:3]
for idx, sample in enumerate(test_samples):
    print(f"\n{'='*60}")
    print(f"TEST SAMPLE {idx + 1}")
    print(f"{'='*60}")
    text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    result = predict_document(
        text=text[:500], 
        words=sample['words'],
        boxes=sample['boxes'],
        entities=sample['entities']
    )
    print(f"\nActual Document Type: {sample['document_type']}")
    print(f"\nPredicted Output:")
    print(json.dumps(result, indent=2))
    print(f"\nExplainability heatmap saved at: {result['explainability_map']}")
    print()