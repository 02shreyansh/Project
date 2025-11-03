import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model, LayoutLMv3Processor, LayoutLMv3Config
from typing import Dict, List, Tuple, Optional
import numpy as np


class MultiModalDocumentAI(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        num_entity_labels: int = 10,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.config = LayoutLMv3Config.from_pretrained(
            "microsoft/layoutlmv3-base", num_labels=num_labels
        )
        self.layoutlm = LayoutLMv3Model.from_pretrained(
            "microsoft/layoutlmv3-base", config=self.config
        )

        self.hidden_size = hidden_size
        self.doc_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_entity_labels),
        )

        self.reasoning_layer = ReasoningLayer(hidden_size)
        self.attention_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        self.attention_weights = outputs.attentions[-1]
        doc_logits = self.doc_classifier(pooled_output)
        entity_logits = self.entity_classifier(sequence_output)
        reasoning_output = self.reasoning_layer(
            sequence_output, attention_mask, entity_logits
        )
        loss = None
        if labels is not None or entity_labels is not None:
            loss_dict = {}

            if labels is not None:
                doc_loss = F.cross_entropy(doc_logits, labels)
                loss_dict["doc_loss"] = doc_loss
        if entity_labels is not None:
            batch_size = entity_logits.size(0)
            seq_len = entity_logits.size(1)
            if entity_labels.size(1) != seq_len:
                if entity_labels.size(1) < seq_len:
                    padding = seq_len - entity_labels.size(1)
                    entity_labels = torch.cat(
                        [
                            entity_labels,
                            torch.full(
                                (batch_size, padding),
                                -100,
                                device=entity_labels.device,
                                dtype=entity_labels.dtype,
                            ),
                        ],
                        dim=1,
                    )
                else:
                    entity_labels = entity_labels[:, :seq_len]

            entity_loss = F.cross_entropy(
                entity_logits.view(-1, entity_logits.size(-1)),
                entity_labels.view(-1),
                ignore_index=-100,
            )
            loss_dict["entity_loss"] = entity_loss

            loss_dict["reasoning_loss"] = reasoning_output["loss"]
            loss = sum(loss_dict.values())

        return {
            "loss": loss,
            "doc_logits": doc_logits,
            "entity_logits": entity_logits,
            "reasoning_output": reasoning_output,
            "attention_weights": self.attention_weights,
            "hidden_states": sequence_output,
        }
