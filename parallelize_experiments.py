from torch.multiprocessing import Pool
from torch.multiprocessing import Process, Queue
import subprocess
import traceback
import argparse
from utils import hmm_default_small_args, rnn_default_small_args, vae_default_small_args
import time
import sys


def run_bash(cmd, queue=None, timeout=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        if stderr:
            stderr = stderr.decode('utf-8')
        else:
            stderr = None
        if stdout:
            stdout = stdout.decode('utf-8')
        else:
            stdout = None
        if queue is not None:
            queue.put((stdout, stderr))
        return stdout, stderr  # This is the stdout from the shell command
    except subprocess.TimeoutExpired:
        if queue is not None:
            queue.put((None, traceback.format_exc()))
        return None, traceback.format_exc()


def task_spooler(commands_lst, num_processes=3, timeout=None):
    run_bash("ts -S {0}".format(num_processes))
    process_lst, std_out_lst, std_err_lst = [], [], []
    for i, command in enumerate(commands_lst):
        ts_command = "ts bash -c '{0}'".format(command)
        std_out, std_err = run_bash(ts_command, timeout=timeout)
        process_lst.append("Process {0}".format(i + 1))
        std_out_lst.append(std_out)
        std_err_lst.append(std_err)
    return process_lst, std_out_lst, std_err_lst


def queue_jobs(commands_lst, num_processes=3, timeout=None):
    # Create several processes, start each one, and collect the results.
    assert(len(commands_lst) % num_processes == 0)
    queue_lst = []
    for i in range(num_processes):
        queue_lst.append(Queue())

    iterations = len(commands_lst) // num_processes
    process_names_lst, std_out_lst, std_err_lst = [], [], []
    for i in range(iterations):
        process_lst = []
        for j, queue in enumerate(queue_lst):
            process_lst.append(Process(target=run_bash, args=(commands_lst[i * num_processes + j], queue, timeout)))
        for process in process_lst:
            process.start()
        for process in process_lst:
            process.join()
        for j, queue in enumerate(queue_lst):
            process_names_lst.append("Process {0}".format(i * num_processes + j + 1))
            std_out, std_err = queue.get()  # you can not request queue.get() twice it is an iterator
            std_out_lst.append(std_out)
            std_err_lst.append(std_err)
            print("=" * 50)
            print("Process {0}".format(i * num_processes + j + 1))
            print("Stdout:", std_out)
            print("Stderr:", std_err)
    return process_names_lst, std_out_lst, std_err_lst


def pool_jobs(commands_lst, num_processes=3, timeout=None):
    # Create several processes, start each one, and collect the results.
    pool = Pool(processes=num_processes)
    results = pool.map(run_bash, commands_lst)
    print(results, len(results))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments for the Model')
    parser.add_argument("-n", "--num_processes", default=3, required=False, help="num processes to run at once", type=int)
    parser.add_argument("-s", "--script", default="./scripts/gfp_small_scripts.sh", required=False, help="script path", type=str)
    args = vars(parser.parse_args())
    num_processes = args["num_processes"]
    script = open(args["script"], "r")

    """ test task spooler """
    # commands_lst = [command.strip() for command in script.readlines()]
    # process_lst, std_out_lst, std_err_lst = task_spooler(commands_lst, num_processes, timeout=None)

    """ test queue jobs """
    commands_lst = [command.strip() for command in script.readlines()]
    process_lst, std_out_lst, std_err_lst = queue_jobs(commands_lst=commands_lst, num_processes=num_processes, timeout=None) # time limit in seconds

    """ test pool jobs """
    # commands_lst = [command.strip() for command in script.readlines()]
    # process_lst, std_out_lst, std_err_lst = pool_jobs(commands_lst=commands_lst, num_processes=num_processes)

    """ test run bash """
    # timeout = 5
    # sleep_time = 1
    # print(run_bash("sleep 3 && echo 'hello' && python -c 'print(1)'", queue=None, timeout=timeout))
    # print(run_bash("python multiprocess_trials.py --time {0}".format(sleep_time), queue=None, timeout=timeout))
