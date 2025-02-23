import os
import shlex
from argparse import ArgumentParser
from collections import defaultdict
from subprocess import PIPE, Popen

import pandas as pd

SAVE_DIR = "data/"


def get_cmd(num_procs, num_updates, nsteps):
    if nsteps is not None:
        nsteps_str = f"habitat_baselines.rl.ppo.num_steps=145"
    else:
        nsteps_str = ""
    # The actual number of environments is 1/6 of the set number (since the RL code treats agent as its own env).
    return f"python -m rl_utils.launcher --cfg /home/aszot/speed.yaml --proj-dat nowb,speed python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/madrona.yaml --run-type train habitat_baselines.num_environments={num_procs*6} habitat_baselines.total_num_steps=-1 habitat_baselines.num_updates={num_updates} {nsteps_str}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-updates", default=10, type=int)
    parser.add_argument("--n-trials", default=3, type=int)
    parser.add_argument("--n-steps", default=None, type=int)
    parser.add_argument("--n-envs", default=3700, type=int)
    parser.add_argument("--name", default="speeds", type=str)
    args = parser.parse_args()

    all_dat = defaultdict(list)

    for proc_scaling in [1, 2, 4, 8, 16, 32]:
        nprocs = args.n_envs // proc_scaling
        cmd = get_cmd(nprocs, args.num_updates, args.n_steps)
        cmd_parts = shlex.split(cmd)
        for _ in range(args.n_trials):
            process = Popen(cmd_parts, stdout=PIPE)
            (output, err) = process.communicate()
            exit_code = process.wait()
            output = output.decode("UTF-8").rstrip()

            log_file = None
            for x in output.split(" "):
                if "habitat_baselines.log_file" in x:
                    log_file = x.split("=")[-1]

            def get_time(line, time_name):
                time = line.split(time_name)[-1].strip().split("\t")[0].strip()
                if "s" in time:
                    time = time[:-1]
                return float(time)

            with open(log_file, "r") as f:
                lines = f.readlines()
                fps = get_time(lines[-3], "fps:") / 6
                env = get_time(lines[-2], "env-time:")
                pth = get_time(lines[-2], "pth-time:")

            print(f"#Procs={nprocs}: fps: {fps}, env: {env}, pth: {pth}")
            all_dat["num-processes"].append(nprocs)
            all_dat["FPS"].append(fps)
            all_dat["sim-time-(seconds)"].append(env)
            all_dat["learning-time-(seconds)"].append(pth)
            if args.debug:
                break
        if args.debug:
            break
    df = pd.DataFrame.from_dict(all_dat)
    print("Final Results\n", df)
    os.makedirs(SAVE_DIR, exist_ok=True)
    df.to_csv(os.path.join(SAVE_DIR, args.name + ".csv"))
