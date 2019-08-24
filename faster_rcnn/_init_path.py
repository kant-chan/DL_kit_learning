import os, sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = os.path.dirname(__file__)

lib_path = os.path.join(root_dir, 'lib')
add_path(lib_path)

data_path = os.path.join(root_dir, 'data', 'coco', 'PythonAPI')
add_path(data_path)
