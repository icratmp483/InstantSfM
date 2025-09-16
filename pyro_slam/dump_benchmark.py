import os
import time
import torch
from datapipes.bal_loader import get_problem, read_bal_data, _problem_lister, _with_base_url
from pyro_slam.sparse.solve import cusolvesp, cudss
import subprocess


DUMP_DIR = 'dump_Ab'

ALL_DATASETS = ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final']

COMPUTE_JTJ = 'csr'
assert COMPUTE_JTJ in ['csr', 'bsr']

if COMPUTE_JTJ == 'bsr':
    from pyro_slam.sparse.py_ops import *
    from pyro_slam.sparse.bsr_cuda import *

total_time = 0.0

for dataset in ALL_DATASETS:
    url_dp = _problem_lister(_with_base_url(dataset + '.html'), cache_dir='tmp_debug')

    for problem in url_dp:
        problem = problem.split(f'https://grail.cs.washington.edu/projects/bal/data/{dataset}/')[-1]
        problem = problem.split('.txt.bz2')[0]

        # load matrix
        sample = f'{DUMP_DIR}/{dataset}_{problem}.pt'
        if not os.path.exists(sample):
            print(f"Skipping {dataset}_{problem}")
            continue
        sample = torch.load(sample)

        A = sample['A'].cuda()
        b = sample['b'].cuda()
        start = time.perf_counter()
        x = cudss(A, b.flatten())
        print((A @ x - b.flatten()))
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_time += end - start
        print(f"Total time: {total_time}")
        continue

        Js, R = sample['J'], sample['R']
        torch.cuda.nvtx.range_push(f"{dataset}_{problem}")
        Jc = Js[0].cuda()
        Jp = Js[1].cuda()
        R = R.cuda()
        torch.cuda.nvtx.range_push("JTJc")

        start = time.perf_counter()
        
        if COMPUTE_JTJ == 'csr':
            Jc = Jc.to_sparse_coo().to_sparse_csc()
            JTJc = torch.matmul(Jc.mT, Jc)
        elif COMPUTE_JTJ == 'bsr':
            JTJc = torch.matmul(Jc.mT, Jc)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("JTJp")
        if COMPUTE_JTJ == 'csr':
            Jp = Jp.to_sparse_coo().to_sparse_csc()
            JTJp = torch.matmul(Jp.mT, Jp)
        elif COMPUTE_JTJ == 'bsr':
            JTJp = torch.matmul(Jp.mT, Jp)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        end = time.perf_counter()
        total_time += end - start

        # torch.cuda.nvtx.range_push("JTR")
        # if COMPUTE_JTJ == 'csr':
        #     JTRc = torch.matmul(Jc.mT, R.flatten())
        #     JTRp = torch.matmul(Jp.mT, R.flatten())
        # elif COMPUTE_JTJ == 'bsr':
        #     ...
        # torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
print(f"Total time: {total_time}")
exit(0)
# Total time: 27.787715041014962 on ROCm
# sudo `which ncu` --kernel-name scan_symbol_kernel --launch-skip 1 --launch-count 1 --set full -o ncurep1n.ncu-rep `which python` dump_benchmark.py