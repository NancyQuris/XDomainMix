import os
from copy import deepcopy
import clip
import random 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.fmow_dataset import FMoWDataset

from .datasets import FMoW_Batched_Dataset

IMG_HEIGHT = 224
NUM_CLASSES = 62
NUM_DOMAINS = 16*5
FEATURE_DIM = 1024

categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]


class Model(nn.Module):
    def __init__(self, args, weights, mixup_module=None, use_clip=False):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        if use_clip:
            self.model, _ = clip.load('ViT-B/16')
            self.classifier = nn.Linear(512, self.num_classes)
            if mixup_module is not None:
                self.mixup_module = mixup_module

            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.enc = densenet121(pretrained=True).features
            self.classifier = nn.Linear(1024, self.num_classes)
            if weights is not None:
                self.load_state_dict(deepcopy(weights))
            if mixup_module is not None:
                self.mixup_module = mixup_module

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        dataset = FMoWDataset(root_dir=args.data_dir, download=True)
        # get all train data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_sets = FMoW_Batched_Dataset(args, dataset, 'train', args.batch_size, transform)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=args.batch_size)
        return train_loaders, tv_loaders

    def clip_zero_shot(self, x):
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(x.device)
        self.text_features = self.model.encode_text(self.text_inputs)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        image_features = self.model.encode_image(x)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self.text_features.T).type(torch.float32)
        return similarity
    
    def clip_train_f(self, x, y, domain, class_gradient, domain_gradient):
        x = self.model.encode_image(x)
        x = self.mixup_module.clip_forward(x, y, domain, class_gradient, domain_gradient)
        x = self.classifier(x)
        return x
    
    def clip_forward(self, x):
        features = self.model.encode_image(x).type(torch.float32)
        out = self.classifier(features)
        return out
    
    def clip_get_feature(self, x):
        features = self.model.encode_image(x).type(torch.float32)
        return features

    def clip_get_result(self, feature):
        out = self.classifier(feature)
        return out 
    
    
    def forward(self, x):
        features = self.enc(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def train_f(self, x, y, domain, class_gradient, domain_gradient):
        x = self.get_feature(x)
        x = self.mixup_module.forward(x, y, domain, class_gradient, domain_gradient)
        x = self.get_result(x)
        return x
    
    def get_feature(self, x):
        x = self.enc(x)
        x = F.relu(x, inplace=True)
        return x

    def get_result(self, feature):
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = torch.flatten(feature, 1)
        out = self.classifier(feature)
        return out 
    
    def get_whole_feature(self, x):
        x = self.get_feature(x)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x 

    def get_prediction(self, feature):
        return self.classifier(feature)

    def train_mixstyle(self, x, mix_module):
        out = x
        out = self.enc[0](out)
        out = self.enc[1](out)
        out = self.enc[2](out)
        out = self.enc[3](out)
        out = self.enc[4](out)
        out = mix_module(out)
        
        out = self.enc[5](out)
        out = self.enc[6](out)
        out = mix_module(out)
        
        out = self.enc[7](out)
        out = self.enc[8](out)
        out = self.enc[9](out)
        out = self.enc[10](out)
        out = self.enc[11](out)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def train_dsu(self, out, pertubration):
        out = self.enc[0](out)
        out = pertubration[0](out)

        out = self.enc[1](out)
        out = self.enc[2](out)
        out = self.enc[3](out)
        out = pertubration[1](out)

        out = self.enc[4](out)
        out = self.enc[5](out)
        out = pertubration[2](out)
        
        out = self.enc[6](out)
        out = self.enc[7](out)
        out = pertubration[3](out)
        
        out = self.enc[8](out)
        out = self.enc[9](out)
        out = pertubration[4](out)
        
        out = self.enc[10](out)
        out = self.enc[11](out)
        out = pertubration[5](out)

        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out