from importlib import import_module


def init_obj(module_name, obj_name, *args, **kwargs):
    """
    Finds a function handle with the name obj_name in module module_name, and returns the
    instance initialized with corresponding arguments given.

    Is equivalent to
    `object = module_name.obj_name(a, b=1)`
    """
    module = import_module(module_name)
    return getattr(module, obj_name)(*args, **kwargs)
