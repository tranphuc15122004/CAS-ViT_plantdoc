import torch
import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
import json
import torch
import numpy as np
import os
import shutil

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer

from data.datasets import build_dataset
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils as utils

from model import *

# Đường dẫn tới file checkpoint
checkpoint_path = r"C:\\PHUC\\CAS-ViT_plantdoc\\casvit_s_ep300.pth"

# Tải checkpoint (map_location='cpu' đảm bảo chạy trên CPU nếu không có GPU)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model = create_model(
        'rcvit_s',
        pretrained=False,
        num_classes= 27,
        drop_path_rate=0.1,
        layer_scale_init_value= 1e-6,
        head_init_scale=1.0,
        input_res=256,
        classifier_dropout=0.3,
        distillation=False,
    )

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
