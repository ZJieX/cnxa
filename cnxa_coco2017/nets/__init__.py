from .resnet50 import resnet50
from .vgg16 import vgg16
from .vit import vit
from .convbk_COCO import convbk_base
from .convbk_COCO_dw import convbk_base_dw
from .convnext import convnext_base
from .convnextaa import convnextaa_base

get_model_from_name = {
    "resnet50": resnet50,
    "vgg16": vgg16,
    "vit": vit,
    "convbk": convbk_base,
    "convbk_dw": convbk_base_dw,
    "convnext_base": convnext_base,
    "convnextaa_base": convnextaa_base,
}
