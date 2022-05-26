import argparse
from pathlib import Path

from attrdict import AttrDict
from yaml import load, CLoader

from src.dss import DssPipeline


def handle_dss(args, conf):
    dss = DssPipeline(conf.dss, args.indir / 'dss', args.outdir)
    dss.run_stage_or_full(args.stage, args.force)


def handle_iam(args, conf):
    print('IAM not implemented yet')


def get_config(args):
    with args.config as file:
        return AttrDict(load(file.read(), Loader=CLoader))


def parse_args():
    # Create top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), default='./config.yaml')
    parser.add_argument('-i', '--indir', type=Path, default='./data')
    parser.add_argument('-o', '--outdir', type=Path, default='./out')
    subparsers = parser.add_subparsers(
        dest='cmd', help='the task to scope to: dss=Dead Sea Scrolls tasks; iam=IAM-dataset task')
    subparsers.required = True

    # Create parser for Dead Sea Scrolls task
    parser_dss = subparsers.add_parser('dss')
    parser_dss.add_argument('-s', '--stage', choices=DssPipeline.STAGES + ['full'], default='full')
    parser_dss.add_argument('-f', '--force', action='store_true')
    # ...dss arguments
    parser_dss.set_defaults(func=handle_dss)

    # Create parses for IAM task
    parser_iam = subparsers.add_parser('iam')
    # ...iam arguments
    parser_iam.set_defaults(func=handle_iam)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args)

    # for debugging purposes
    if args.debug:
        from pprint import pprint
        print('args:')
        pprint(args)
        print('\nconfig:')
        pprint(conf)

    out_dir = args.outdir.resolve()

    # In case of emergency, uncomment the line below
    # from time import sleep; print("Doing work..."); sleep(10); print("Final performance: 100%")); import sys; sys.exit(0)

    args.func(args, conf)
