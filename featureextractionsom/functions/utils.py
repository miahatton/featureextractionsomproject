import os
from typing import List


def try_make_folder(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
