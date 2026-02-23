import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

class BiLSTMPooling(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, token_embeddings):
        lstm_out, _ = self.lstm(token_embeddings)
        return lstm_out.mean(dim=1)
    
class SentimentClassifier(nn.Module):
    def __init__(self, num_labels, num_semantic_classes, num_semantic_slots, semantic_emb_dim, lstm_hidden_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased") # вынести
        for param in self.bert.parameters():
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
        self.drop = nn.Dropout(p=0.3)
        self.semantic_class_embedding = nn.Embedding(num_semantic_classes, semantic_emb_dim)
        self.semantic_slot_embedding = nn.Embedding(num_semantic_slots, semantic_emb_dim)
        self.semantic_lstm = BiLSTMPooling(emb_dim=semantic_emb_dim, hidden_size=lstm_hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 2 * 2 * lstm_hidden_size, num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, slots=None, classes=None, labels=None):
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=False)

        mask = attention_mask.unsqueeze(-1).float()
        semantic_class_embeds = self.semantic_class_embedding(classes)  
        semantic_slot_embeds = self.semantic_slot_embedding(slots)    
        semantic_class_embeds = semantic_class_embeds * mask
        semantic_slot_embeds = semantic_slot_embeds * mask
        class_summary = self.semantic_lstm(semantic_class_embeds) 
        slot_summary = self.semantic_lstm(semantic_slot_embeds)   
        combined = torch.cat([pooled_output, class_summary, slot_summary], dim=1)
        logits = self.classifier(combined)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
        
    def save_pretrained(self):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            if not value.is_contiguous():
                state_dict[key] = value.contiguous()
        torch.save(state_dict, "./pytorch_model.bin")