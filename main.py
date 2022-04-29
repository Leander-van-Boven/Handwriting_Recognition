import argparse

from dss import classifier as dss_classifier, segmenter as dss_segmenter


def dss(args):
    classifier = dss_classifier(args.conf)
    segmenter = dss_segmenter(args.conf)


def iam(args):
    pass


def parse_args():
    # Create top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='the task to scope to: dss=Dead Sea Scrolls tasks; iam=IAM-dataset task')

    # Create parser for Dead Sea Scrolls task
    parser_dss = subparsers.add_parser('dss')
    # ...arguments
    parser_dss.set_defaults(func=dss)

    # Create parses for IAM task
    parser_iam = subparsers.add_parser('iam')
    # ...arguments
    parser_iam.set_defaults(func=iam)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.func(args)
