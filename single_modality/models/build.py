from timm.models import create_model
from . import videofocalnet

def build_model(model_type,is_pretrained,num_classes):
    print(f"Creating model: {model_type}")
    
    if "focal" in model_type:
        model = create_model(
            model_type, 
            pretrained=is_pretrained,
            num_classes=num_classes
        )                      
    elif "vit" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config.DATA.IMG_SIZE,
            num_classes=num_classes,
        )
    elif "resnet" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=num_classes
        )
    else:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=num_classes
        )        
    return model