from attrdict import AttrDict

from src.factory_base import FactoryBase


def classifier_factory(conf: AttrDict):
    return FactoryBase('src.dss.classification', conf)


def segmenter_factory(conf: AttrDict):
    return FactoryBase('src.dss.segmentation', conf)
