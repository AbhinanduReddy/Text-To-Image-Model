from transformers import BertTokenizer, BertModel
import torch
from torch import nn

class ViLT(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased'):
        super().__init__()
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.image_fc = nn.Linear(2048, 768)  # Assuming 2048-dim visual features
        self.transformer = nn.Transformer(d_model=768, nhead=8, num_encoder_layers=6)
        
    def forward(self, visual_features, input_ids, attention_mask):
        visual_embeds = self.image_fc(visual_features).unsqueeze(1)
        text_embeds = self.text_model(input_ids, attention_mask).last_hidden_state
        combined_embeds = torch.cat((visual_embeds, text_embeds), dim=1)
        combined_embeds = self.transformer(combined_embeds, combined_embeds)
        return combined_embeds[:, 0, :]  # Return the CLS token

