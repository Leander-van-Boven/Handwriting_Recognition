from attrdict import AttrDict

from .base import ImplementationFinder


def classifier(conf: AttrDict):
    return ImplementationFinder('dss.classification', conf)


def segmenter(conf: AttrDict):
    return ImplementationFinder('dss.segmentation', conf)
