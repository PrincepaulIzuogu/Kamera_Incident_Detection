import sys
sys.path.append('/openpose/python')

from openpose import pyopenpose as op

def setup_openpose():
    params = {
        "model_folder": "/openpose/models/",
        "hand": False,
        "face": False,
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper
