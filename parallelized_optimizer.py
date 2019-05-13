from torch.multiprocessing import Pool
from torch.multiprocessing import Process, Queue
import subprocess
import traceback
import argparse
from utils import hmm_default_small_args, rnn_default_small_args, vae_default_small_args
import time
import sys
import json
import pandas as pd
import numpy as np


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


def queue_jobs(commands_lst, skip = 0, num_processes=4, timeout=None, name="sample"):
    # Create several processes, start each one, and collect the results.
    assert(len(commands_lst) % num_processes == 0)
    start_time = time.time()
    queue_lst = []
    for i in range(num_processes):
        queue_lst.append(Queue())

    iterations = len(commands_lst) // num_processes
    process_names_lst, std_out_lst, std_err_lst = [], [], []
    columns = ['model_type', 'test_score', 'total_time', 'dataset', 'average_mismatches', 'valid_score', 'train_score', 
               'num_params', 'exit_code', 'name']
    std_out_df = pd.DataFrame(columns=columns)
    std_err_lst = []
    for i in range(iterations):
        if i < skip // num_processes: 
            continue
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
            if std_out:
                string_dict = std_out.split('\n')[-2]
                std_out_df = std_out_df.append(eval(string_dict), ignore_index=True)
            else: 
                std_out_df = std_out_df.append(dict(zip(columns, np.zeros(len(columns)))), ignore_index=True)
            print("=" * 50)
            print("Process {0}".format(i * num_processes + j + 1))
            print("Stdout:", std_out)
            print("Stderr:", std_err)
        np.save("./logs/optimized_results/{0}_sample_std_out.npy".format(name), np.array(std_out_lst))
        np.save("./logs/optimized_results/{0}_sample_std_err.npy".format(name), np.array(std_err_lst))
        std_out_df.to_csv("./logs/optimized_results/{0}_sample_std_out.csv".format(name), index=False)
        print("Ran {0}/{1} experiments in {2:.4f} seconds".format((i + 1) * num_processes, iterations * num_processes, time.time() - start_time))
    return process_names_lst, std_out_lst, std_err_lst, std_out_df, time.time() - start_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments for the Model')
    parser.add_argument("-n", "--num_processes", default=4, required=False, help="num processes to run at once", type=int)
    parser.add_argument("-s", "--script", default="./scripts/gfp_small_scripts.sh", required=False, help="script path", type=str)
    parser.add_argument("-na", "--name", default="small_optimize", required=False, help="optimization name", type=str)
    parser.add_argument("-t", "--time_out", default=None, required=False, help="time out in seconds", type=int)
    parser.add_argument("-sk", "--skip", default=0, required=False, help="skip which experiments", type=int)
    args = vars(parser.parse_args())
    num_processes = args["num_processes"]
    script = open(args["script"], "r")
    name = args["name"]
    """ test queue jobs """
    commands_lst = [command.strip() for command in script.readlines()]
    std_out_df, process_lst, std_out_lst, std_err_lst, total_time = queue_jobs(commands_lst=commands_lst,skip=args["skip"], num_processes=num_processes,timeout=args["time_out"], name=name)
    
    
