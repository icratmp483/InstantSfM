from datapipes.bal_loader import get_problem, read_bal_data, _problem_lister, _with_base_url
import subprocess

ALL_DATASETS = ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final']
for dataset in ALL_DATASETS:
    url_dp = _problem_lister(_with_base_url(dataset + '.html'), cache_dir='tmp_debug')

    for problem in url_dp:
        problem = problem.split(f'https://grail.cs.washington.edu/projects/bal/data/{dataset}/')[-1]
        problem = problem.split('.txt.bz2')[0]
        print(problem)

        # launch python script
        subprocess.run(['python', 'dump_Ab.py', dataset, problem])
