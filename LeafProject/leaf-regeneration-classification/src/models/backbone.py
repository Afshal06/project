class Backbone:
    def __init__(self, model_name='resnet50', pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = self._load_model()

    def _load_model(self):
        if self.model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=self.pretrained)
            model = self._modify_model(model)
            return model
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def _modify_model(self, model):
        # Modify the model for our specific use case
        num_classes = 10  # Example: number of leaf classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x):
        return self.model(x)