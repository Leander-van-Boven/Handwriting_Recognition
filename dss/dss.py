from attrdict import AttrDict

from .base import FactoryBase


def classifier_factory(conf: AttrDict):
    return FactoryBase('dss.classification', conf)


def segmenter_factory(conf: AttrDict):
    return FactoryBase('dss.segmentation', conf)
