from timm.models import create_model
from . import videofocalnet

def build_model(model_type,is_pretrained):
    print(f"Creating model: {model_type}")
    
    if "focal" in model_type:
        model = create_model(
            model_type, 
            pretrained=is_pretrained
        )                      
    elif "vit" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif "resnet" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )
    else:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )        
    return model