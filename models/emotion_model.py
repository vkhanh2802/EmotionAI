import torch
import torch.nn as nn
from models.backbone import ResNet50Backbone
from models.classifier import EmotionHead

class FacialAffectModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialAffectModel, self).__init__()

        self.backbone = ResNet50Backbone()

        self.emotion_classifier = EmotionHead(in_features=2048, num_classes=num_classes)
        
        # Không gian chờ cho Multi-task Learning (Giai đoạn sau)
        # self.va_classifier = VAHead(...)
        # self.au_classifier = AUHead(...)

    def forward(self, x):

        features = self.backbone(x)
        

        emotion_logits = self.emotion_classifier(features)
        
        # Sau này sẽ nâng cấp thành: return emotion_logits, va_logits, au_logits
        return emotion_logits
    