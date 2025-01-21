from  . import transforms as T
from typing import List, Dict

def build_transform(params: List[Dict]):
    transform_list = []
    for param in params:
        fn = getattr(T, param.pop("type"))
        transform_list.append(fn(**param))
        


    return T.Compose(transform_list)
