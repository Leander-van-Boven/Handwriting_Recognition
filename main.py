import argparse
from pathlib import Path

from attrdict import AttrDict
from yaml import load, CLoader

from src.dss import DssPipeline
from src.iam import IamPipeline


def handle_dss(args, conf):
    dss = DssPipeline(conf.dss, args)
    dss.run_stage_or_full(args.stage)


def handle_iam(args, conf):
    iam = IamPipeline(conf.iam, args)
    iam.run_stage_or_full(args.stage)


def get_config(args):
    with args.config as file:
        return AttrDict(load(file.read(), Loader=CLoader))


def parse_args():
    # Create top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='print the config and parsed arguments')
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), default='./config.yaml', metavar='PATH',
                        help='the location of the config yaml file')
    parser.add_argument('-o', '--outdir', type=Path, default='./out', metavar='PATH',
                        help='the directory to store the generated output files')
    parser.add_argument('-s', '--save-intermediate', action='store_true', help='save the output of all '
                                                                               'intermediate steps to disk')
    parser.add_argument('-l', '--load-intermediate', action='store_true', help='load all intermediate steps'
                                                                               ' from disk, when available')
    subparsers = parser.add_subparsers(
        dest='cmd', help='the task to scope to: dss=Dead Sea Scrolls tasks; iam=IAM-dataset task')
    subparsers.required = True

    # Create parser for Dead Sea Scrolls task
    parser_dss = subparsers.add_parser('dss')
    parser_dss.add_argument('-i', '--indir', type=Path, default='./data/dss', metavar='PATH',
                            help='the directory from which to load the input dataset')
    parser_dss.add_argument('-S', '--stage', choices=DssPipeline.STAGES + ['full'], default='full', metavar='OPT',
                            help='the stage to execute, full equals to the entire inference pipeline')
    parser_dss.add_argument('-g', '--glob', type=str, default='scrolls/*binarized.jpg', help='the file pattern used to '
                                                                                             'determine the images '
                                                                                             'used as input')
    # ...dss arguments
    parser_dss.set_defaults(func=handle_dss)

    # Create parses for IAM task
    parser_iam = subparsers.add_parser('iam')
    parser_iam.add_argument('-i', '--indir', type=Path, default='./data/iam', metavar='PATH',
                            help='the directory from which to load the input dataset')
    parser_iam.add_argument('-S', '--stage', choices=IamPipeline.STAGES + ['full'], default='full', metavar='OPT',
                            help='the stage to execute: ' + ', '.join(IamPipeline.STAGES + ['full']))
    parser_iam.add_argument('-g', '--glob', type=str, default='img/*.png', help='the file pattern used to '
                                                                                'determine the images '
                                                                                'used as input')
    # ...iam arguments
    parser_iam.set_defaults(func=handle_iam)

    return parser.parse_args()


def main():
    args = parse_args()
    conf = get_config(args)

    # for debugging purposes
    if args.debug:
        from pprint import pprint
        print('args:')
        pprint(args)
        print('\nconfig:')
        pprint(conf)
        input('continue?')

    args.func(args, conf)


if __name__ == '__main__':
    # In case of emergency, uncomment the line below
    # from time import sleep; print("Doing work..."); sleep(10); print("Final performance: 100%")); return
    main()
