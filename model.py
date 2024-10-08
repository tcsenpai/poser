from transformers import TFViTForImageClassification, ViTConfig, ViTFeatureExtractor

class PostureNet:
    def __init__(self):
        self.model = None
        self.feature_extractor = None

    def build_model(self, num_labels=2):
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=num_labels)
        self.model = TFViTForImageClassification(config)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        return self.model

    def load_model(self, model_path):
        self.model = TFViTForImageClassification.from_pretrained(model_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        return self.model

    def preprocess_input(self, images):
        return self.feature_extractor(images=images, return_tensors="tf")
