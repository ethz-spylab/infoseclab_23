import inspect
import os

def get_checkpoint_abs_path(rel_path):
    """Converts relative path of checkpoint-related resource to absolute path.

    Args:
        rel_path: path of the file relative to repository root.

    Returns:
        Absolute path of the file.

    Let's say that tutorials are located in `/path/to/tutorials/` directory,
    which means that full path of this file is `/path/to/tutorials/common/defense.py`.
    Then following call to this method:
        `get_checkpoint_abs_path('checkpoints/model-1')`
    will return `/path/to/tutorials/checkpoints/model-1`
    """
    module_filename = inspect.getfile(inspect.currentframe())
    module_dirname = os.path.dirname(os.path.abspath(module_filename))
    tutorials_root = os.path.abspath(os.path.join(module_dirname, '..'))
    return os.path.join(tutorials_root, rel_path)

