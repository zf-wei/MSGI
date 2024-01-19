import argparse
import numpy as np
import h5py
from Size_and_Dim_001 import OneRound


parser = argparse.ArgumentParser(description="Run the program with command-line arguments.")
parser.add_argument("--method_id", type=int, help="Method ID", required=True)
parser.add_argument("--itera", type=int, help="Number of iterations", default=5)

args = parser.parse_args()

RECORD_parallel = OneRound(args.method_id, args.itera)
RECORD_parallel = np.array(RECORD_parallel)

with h5py.File(f'{args.method_id}.record', 'w') as hf:
    hf.create_dataset('RECORD_parallel', data=RECORD_parallel)

for i in [0, 1, 2, 3]:
    np.savetxt(f"{args.method_id}_{i}.mean", np.mean(RECORD_parallel, axis=0)[:, :, i])