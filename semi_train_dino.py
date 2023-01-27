
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import copy
import pandas as pd
from PIL import Image
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

import lightly
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.utils import BenchmarkModule
from lightly.models import utils
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.modules import heads
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
import gc

gc.collect()

src = "/label"
dataset_csv = "dataset_csv"


torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()




# Settings for Caltech 256
img_size = 150
batch_size = 256
num_workers = 8
max_epochs = 200
num_classes = 256
output_dim = 2048

lr_factor = batch_size / 128 # scales the learning rate linearly with batch size
knn_k = 20
knn_t = 0.1




# Multi crop augmentation for DINO, additionally, disable blur for cifar10
collate_fn = lightly.data.DINOCollateFunction(
                    global_crop_size = 150
                    )



class OfficeDataset(torch.utils.data.Dataset):
    def __init__(self, img_csv_filepath, label_csv_filepath, transform=None):
        self.img_csv_file = pd.read_csv(img_csv_filepath)
        self.label_csv_file = pd.read_csv(label_csv_filepath)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_csv_file)

    def __getitem__(self, idx):
        file_det = self.img_csv_file.iloc[idx]
        image = Image.open(file_det["path"])
        label = file_det["label"]
        label = self.label_csv_file[self.label_csv_file["label"] == label].values[0][1]
        # label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
            
        return (image, label, file_det["filename"])


# Label encodings
# label_csv = join(src, dataset_csv, "office_31_labels.csv")
label_csv = join(src, dataset_csv, "caltech_256_labels.csv")


# train_csv = join(src, dataset_csv, "paperspace_caltech_256_train.csv")
# train_csv = join(src, dataset_csv, "paperspace_modern_office_31_new_amazon.csv")
# train_csv = join(src, dataset_csv, "paperspace_office_31_dslr.csv")
train_csv = join(src, dataset_csv, "paperspace_caltech_256_train.csv")
train_dataset = OfficeDataset(img_csv_filepath = train_csv,
                              label_csv_filepath = label_csv)

# train_dataset = LightlyDataset("modern_office_31/dslr")

train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn = collate_fn,
                              drop_last=True,
                              num_workers=num_workers)

# ------------------------------------------------------



test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

classif_train_dataset = OfficeDataset(img_csv_filepath = train_csv,
                              label_csv_filepath = label_csv,
                              transform = test_transforms)

# classif_train_dataset = LightlyDataset("office_31/dslr",
#                                        transform = test_transforms)

# classif_train_dataset =  torchvision.datasets.ImageFolder("modern_office_31/amazon",
#                                                            transform = classif_train_transforms)

classif_train_dataloader = DataLoader(classif_train_dataset, 
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

# ------------------------------------------------------



steps_per_epoch = len(train_dataset) // batch_size


    
class DINOModel(BenchmarkModule):
    def __init__(self, backbone, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
        # create a ResNet backbone and remove the classification head
        
        feature_dim = list(backbone.children())[-1].in_features
        # self.backbone = nn.Sequential(
        #     *list(backone.children())[:-1],
        #     nn.AdaptiveAvgPool2d(1)
        # )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = heads.DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = lightly.loss.DINOLoss(output_dim=output_dim)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.999)
        utils.update_momentum(self.head, self.teacher_head, m=0.999)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        param = list(self.backbone.parameters()) \
            + list(self.head.parameters())
        optim = LARS(
            param,
            lr=0.3 * lr_factor,
            weight_decay=1e-6,
            momentum=0.9,
        )
        scheduler = {
            "scheduler": LambdaLR(
                optimizer=optim,
                lr_lambda=linear_warmup_decay(
                    warmup_steps=steps_per_epoch * 10, 
                    total_steps=steps_per_epoch * max_epochs, 
                    cosine=True,
                )
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]

    def training_epoch_end(self, outputs):
        pass
    


### Setting up the encoder -----------------


base_model = torchvision.models.resnet50()
model_path = "models/resnet50-19c8e357.pth"


if model_path is not None:
    base_model.load_state_dict(torch.load(join(src,model_path)), strict=False)
    print("Model weights loaded")

model = DINOModel(base_model, train_dataloader, num_classes)

### Train the encoder as per DINO ---------------
trainer = pl.Trainer(
    max_epochs=max_epochs, 
    gpus=1)

trainer.fit(model, train_dataloaders=train_dataloader)




### Save the encoder weights --------------------

pretrained_model_backbone = model.backbone

# you can also store the backbone and use it in another code
state_dict = {
    'model_params': pretrained_model_backbone.state_dict(),
    'epochs' : max_epochs,
    'output_dim' : output_dim,
    'batch_size' : batch_size,
    'img_size' : img_size,
    'color_augs' : True,
    'lr' : 0.3,
    "random_init" : False
}
torch.save(state_dict, 'res50_dino_caltech.pth')