# -*- coding: utf-8 -*-

import multiprocessing
from .simulation import run
from .simulation_no_visual import run_bg
import json
import time


def main(config):
    run(config)

def main_bg(config, index, return_dict):
    deleted_boids = run_bg(config, index)
    return_dict[index] = deleted_boids


if __name__ == "__main__":
    config = {  "n_boids": 45,
                "n_predators": 3,
                "n_obstacles": 4,
                "max_size": 35,
                "domain": [4000,4000],
                "dt": 0.08,
                "deleted_boids_max": 3}
    n_runs = 4 # only one run possibel with visual output
    all_deleted_boids = []
    name = "Run7.json" # name of the json file
    visual_output = True

    # multiprocessing stuff
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes =[]

    if visual_output:
        # run with visual output wont stop until esc is pressed
        # domain will depend on the screen size
        main(config)
    else:
        pool = multiprocessing.Pool(processes=4)
        for run_step in range(n_runs):
            pool.apply_async(main_bg, args=[config,run_step,return_dict])
        pool.close()
        pool.join()

        #Access returns
        for key in range(n_runs):
            all_deleted_boids.extend(return_dict[key])

    with open('./boids/output/'+name, 'w') as f:
        json.dump(all_deleted_boids, f)
