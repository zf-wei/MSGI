import argparse
import numpy as np
import h5py
from Size_and_Dim import MoreRound

def run_SD(method_id: int, itera: int, num_workers: int):
    RECORD_parallel = MoreRound(method_id, itera, num_workers)
    RECORD_parallel = np.array(RECORD_parallel)

    with h5py.File(f'{method_id}.record', 'w') as hf:
        hf.create_dataset('RECORD_parallel', data=RECORD_parallel)

    for i in [0, 1, 2, 3]:
        np.savetxt(f"{method_id}_{i}.mean", np.mean(RECORD_parallel, axis=0)[:, :, i])


parser = argparse.ArgumentParser(description="Run the program with command-line arguments.")
parser.add_argument("--method_id", type=int, help="Method ID", required=True)
parser.add_argument("--itera", type=int, help="Number of iterations", default=5)
parser.add_argument("--num_workers", type=int, help="Number of workers", default=10)

args = parser.parse_args()

run_SD(args.method_id, args.itera, args.num_workers)
