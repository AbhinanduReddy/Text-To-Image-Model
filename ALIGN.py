import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import timm

class ImageEncoder(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool='avg')

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pretrained=True):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]

class ALIGNModel(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.temperature = temperature

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        logits = (text_features @ image_features.T) / self.temperature
        return logits

# Implement the training loop.
