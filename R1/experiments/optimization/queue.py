"""Run resumable experiment groups with local logs and bounded failure retries."""
import argparse
import json
import subprocess
import sys
import time
import os
if os.name=='nt':
    import msvcrt
else:
    import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from fedperg.paths import LOGS, prepare_output_directories

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    'gpu_features': [
        ['--suite', 'natural', '--device', 'cuda'],
        ['--suite', 'matrix', '--device', 'cuda'],
    ],
    'deep_evidence': [['--suite', 'deep', '--methods', 'FedPERG', 'Router', '--device', 'cuda']],
    'deep_controls': [['--suite', 'deep', '--methods', 'RandomInput', 'FedAvgM-Gate', '--device', 'cuda']],
    'gpu_attribution': [['--suite', 'attribution', '--device', 'cuda']],
    'gpu_robustness': [
        ['--suite', 'stress', '--device', 'cuda'],
        ['--suite', 'quality', '--methods', 'FedPERG', '--device', 'cuda'],
        ['--suite', 'sensitivity', '--device', 'cuda'],
        ['--suite', 'calibration', '--device', 'cuda'],
        ['--suite', 'shift', '--device', 'cuda'],
    ],
    'regimes_reward': [
        ['--suite', 'attribution', '--regimes', 'label_skew', 'quantity_skew',
         '--seeds', '220', '221', '222', '--methods', 'FedPERG', 'Router', 'SpatialOnly',
         'TemporalOnly', 'NoResidual', '--device', 'cuda'],
        ['--suite', 'reward', '--device', 'cuda'],
    ],
    'quality_cnn': [
        ['--suite', 'quality', '--methods', 'Router', 'TemporalOnly', '--device', 'cuda'],
        ['--suite', 'cnn', '--device', 'cuda'],
    ],
    'timing': [['--suite', 'timing', '--seeds', '220', '221', '222', '--device', 'cuda']],
    'calibration_extra': [
        ['--suite','calibration','--device','cuda'],
        ['--suite','shift','--device','cuda'],
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=(*GROUPS, 'all'), required=True)
    args = parser.parse_args()
    prepare_output_directories()
    if args.group == 'all':
        groups = ['deep_evidence', 'deep_controls', 'gpu_features', 'gpu_attribution',
                  'gpu_robustness', 'regimes_reward', 'quality_cnn']
        def run_group(group):
            command = [sys.executable, '-u', '-m', 'experiments.optimization.queue', '--group', group]
            print(f'START GPU GROUP {group}', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            print(f'COMPLETE GPU GROUP {group}', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(run_group, group) for group in groups]
            for task in as_completed(pending):
                task.result()
        run_group('timing')
        print('All GPU groups and isolated timing completed', flush=True)
        return
    # An explicit group and the coordinator may meet on resume. The OS lock
    # prevents duplicate writers and releases automatically if a process exits.
    lock_handle = (LOGS / f'{args.group}.lock').open('a+b')
    if lock_handle.tell() == 0:
        lock_handle.write(b'0')
        lock_handle.flush()
    while True:
        try:
            lock_handle.seek(0)
            if os.name=='nt':msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:fcntl.flock(lock_handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except OSError:
            time.sleep(1)
    state_path = LOGS / f'r1_opt_queue_{args.group}.json'
    state = {'group': args.group, 'status': 'running', 'completed_commands': 0}

    def save():
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')

    save()
    log_path = LOGS / f'r1_opt_queue_{args.group}.log'
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        for options in GROUPS[args.group]:
            command = [sys.executable, '-u', '-m', 'experiments.optimization.confirm', *options]
            state['current_command'] = command
            save()
            for attempt in range(3):
                log.write('\nSTART ' + subprocess.list2cmdline(command) + '\n')
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    break
                state['last_exit_code'] = result.returncode
                state['attempt'] = attempt + 1
                save()
                time.sleep(5)
            else:
                state['status'] = 'failed'
                save()
                raise SystemExit(result.returncode)
            state['completed_commands'] += 1
            save()
    state['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
