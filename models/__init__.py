import glob
import os.path as osp
from utils.common_utils import load_module_from_file


def get_backbone(backbone_name, backbone_kwargs):
    model_folder = osp.dirname(osp.abspath(__file__))
    backbone_folder = osp.join(model_folder, 'backbones')

    backbone_filenames = sorted(glob.glob(osp.join(backbone_folder, '*.py')))

    backbone_modules = []
    for backbone_filename in backbone_filenames:
        backbone_modules.append(load_module_from_file(backbone_filename))

    found = False
    for module in backbone_modules:
        module_cls = getattr(module, backbone_name, None)
        if module_cls:
            found = True
            backbone = module_cls(**backbone_kwargs)
            break

    if not found:
        raise NotImplementedError(f'Unrecognized backbone {backbone_name}')

    return backbone


def get_model(model_name, model_kwargs):
    model_folder = osp.dirname(osp.abspath(__file__))

    model_filenames = sorted(glob.glob(osp.join(model_folder, '*_model.py')))

    model_modules = []
    for model_filename in model_filenames:
        model_modules.append(load_module_from_file(model_filename))

    found = False
    for module in model_modules:
        module_cls = getattr(module, model_name, None)
        if module_cls:
            found = True
            model = module_cls(**model_kwargs)
            break

    if not found:
        raise NotImplementedError(f'Unrecognized {model_name} model!')

    return model
