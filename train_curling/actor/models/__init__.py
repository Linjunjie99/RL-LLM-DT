from core.registry import Registry

model_registry = Registry('Model')

from models.ac_model import *
from models.ac_con_model import *
from models.custom_model import *
from models.q_model import *


def get_model(args):
    """Initialize model instances"""
    model_cls = model_registry.get(args.model)
    model_instances = []

    def create_model_instance():
        model_instances.append(model_cls((20), (29)))

    if args.model_config is not None and 'model' in args.model_config:
        model_config = args.model_config['model']

        if isinstance(model_config, list):
            for _ in enumerate(model_config):
                create_model_instance()
        elif isinstance(model_config, dict):
            create_model_instance()

    else:
        create_model_instance()
        

    if len(model_instances) == 1:
        return model_instances[0]
    else:
        return model_instances
