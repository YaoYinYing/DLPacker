#!/usr/bin/env python

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from DLPacker.utils import convert_keras_h5_to_pt


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--weights-prefix', required=True, help='Prefix without extension, e.g. /path/DLPacker_weights')
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--nres', type=int, default=6)
parser.add_argument('--grid-size', type=int, default=40)
parser.add_argument('--num-channels', type=int, default=27)
args = parser.parse_args()

h5_path = f'{args.weights_prefix}.h5'
out_pt = f'{args.weights_prefix}.pt'

convert_keras_h5_to_pt(
    keras_h5_path=h5_path,
    out_pt_path=out_pt,
    width=args.width,
    nres=args.nres,
    grid_size=args.grid_size,
    num_channels=args.num_channels,
)

print(f'Wrote: {out_pt}')
