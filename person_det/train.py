import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import albumentations as A
import person_trainer as Trainer
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# load a model pre-trained on COCO
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

class_names = ["person"]
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)


# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(
    model,
    criterion,
    optimizer,
    exp_lr_scheduler,
    train_dataloader,
    val_dataloader,
    cuda=True,
    early_stopping_patience=early_stopping_patience,
    checkpoint_path=checkpoint_path,
    class_names=class_names,
)

# trainer.train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders,
#         dataset_sizes, device = "cuda:0", num_epochs=25)
# # go, go, go... call fit on trainer
res = trainer.fit(epochs)
