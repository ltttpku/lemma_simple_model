# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
from transformers import BertTokenizer, VisualBertModel
import torch
import torch.nn as nn
import torchvision

def build_resnet(model_name, pretrained=False):
    cnn = getattr(torchvision.models, model_name)(pretrained=pretrained)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    return model

class VisualBERT(nn.Module):
    def __init__(self, BertTokenizer_CKPT="bert-base-uncased", VisualBertModel_CKPT="uclanlp/visualbert-vqa-coco-pre", output_dim=100) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BertTokenizer_CKPT)
        self.model = VisualBertModel.from_pretrained(VisualBertModel_CKPT)
        self.classifier = nn.Linear(768, output_dim)
        self.output_dim = output_dim

    def forward(self, sentences, visual_embeds):
        '''
        params: sentences: list of sentences in strings
        '''
        inputs = self.tokenizer(sentences, return_tensors="pt")
        # visual_embeds = torch.rand(1, 20, 2048)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float) 
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        logits = self.classifier(last_hidden_states[:, 0, :])
        return logits # # B x output_dim


if __name__ == '__main__':
    model = VisualBERT()
    sentences = ['the people\'s', 'hi an apple?', 'what is this?']
    visual_embeds = torch.rand(3, 20, 2048)
    logits = model(sentences, visual_embeds)
    print(logits.shape)