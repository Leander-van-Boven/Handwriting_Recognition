from attrdict import AttrDict

from .base import StageBase


def classifier(conf: AttrDict):
    return StageBase('dss.classification', conf)


def segmenter(conf: AttrDict):
    return StageBase('dss.segmentation', conf)
