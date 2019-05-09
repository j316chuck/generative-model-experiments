from multiprocessing import Pool
from multiprocessing import Process, Queue
import subprocess
from utils import hmm_default_small_args, rnn_default_small_args, vae_default_small_args


def run_bash(cmd, queue=None):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    std_out = p.stdout.read().strip()
    if p.stderr:
        std_err = p.stderr.read().strip()
    else:
        std_err = None
    if queue is not None:
        queue.put(std_out)
    return std_out, std_err  # This is the stdout from the shell command


def task_spooler(commands_lst, num_processes=5):
    run_bash("ts -S {0}".format(num_processes))
    std_out_lst, std_err_lst = [], []
    for command in commands_lst:
        ts_command = "ts bash -c '{0}'".format(command)
        std_out, std_err = run_bash(ts_command)
        std_out_lst.append(std_out)
        std_err_lst.append(std_err)
    for std_out, std_err in zip(std_out_lst, std_err_lst):
        print(std_out, std_err) # i don't think printing out does much
    return std_out_lst, std_err_lst


def queue_jobs(commands_lst, num_processes=4):
    # Create several processes, start each one, and collect the results.
    assert(len(commands_lst) % num_processes == 0)
    queue_lst = []
    for i in range(num_processes):
        queue_lst.append(Queue())

    iterations = len(commands_lst) // num_processes
    outputs_lst = []
    for i in range(iterations):
        process_lst = []
        for j, queue in enumerate(queue_lst):
            process_lst.append(Process(target=run_bash, args=(commands_lst[i * num_processes + j], queue)))
        for process in process_lst:
            process.start()
        for process in process_lst:
            process.join()
        for j, queue in enumerate(queue_lst):
            outputs_lst.append(("Process {0}".format(i * num_processes + j), queue.get()))
    for output in outputs_lst:
        print(output[0])
        print(output[1])
    return outputs_lst


def pool_jobs(commands_lst, num_processes=4):
    # Create several processes, start each one, and collect the results.
    pool = Pool(processes=num_processes)
    results = pool.map(run_bash, commands_lst)
    print(results, len(results))
    return results


if __name__ == '__main__':
    num_processes = 3
    script = open("./scripts/synthetic_unimodal_small_script.sh", "r")
    commands_lst = [command.strip() for command in script.readlines()]
    task_spooler(commands_lst=commands_lst, num_processes=num_processes)
    # queue_jobs(commands_lst=commands_lst, num_processes=num_processes)
    # pool_jobs(commands_lst=commands_lst, num_processes=num_processes)



