import pickle
import argparse

parser = argparse.ArgumentParser('convert pickle files to contain only standard objects')
parser.add_argument('input', help='Input pickle file')
parser.add_argument('output', help='Input pickle file')
args = parser.parse_args()
with open(args.input, 'rb') as f:
    tiled_v, params = pickle.load(f)
    output = {
        'params': params,
        'value_functions': []
    }
    for v in tiled_v._vs:
        output['value_functions'].append({
             'resolution': v.resolution,
             'lower_limits': v.lower_limits,
             'upper_limits': v.upper_limits,
             'values': v._v
        })
with open(args.output, 'wb') as f:
    pickle.dump(output, f)
