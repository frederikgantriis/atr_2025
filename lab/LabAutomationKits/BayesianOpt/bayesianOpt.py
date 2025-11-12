import random
import evaluateFitness
from customCallbacks import PlotterCallback
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import keyboard

from skopt import gp_minimize
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import load
from skopt.plots import plot_convergence
from skopt.plots import plot_gaussian_process

class Tracker(object):
    step = 0

tracker = Tracker()


pause = False
np.random.seed(777)
noise_level = 0.1
step = 0

bestFitness = 100000000000

def toggle_pause(exception):
    global pause
    pause = not pause
    if pause:
        print("Paused!")
    else:
        print("Unpaused!")

keyboard.on_press_key("f12", toggle_pause)

def evaluate(individual):
    global pause
    global bestFitness
    #print(tracker.step,tracker.step)
    while pause:
        pass
    fitness = evaluateFitness.evaluate(individual,tracker.step,tracker.step)
    print(individual,fitness)
    if fitness < bestFitness:
        bestFitness = fitness
    return fitness





def main(checkPoint=None):

    # Problem size
    N = 3 #Sample dimension
    NSamples = 20 #Max number of samples
    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9) # keyword arguments will be passed to `skopt.dump`

    
    
    if checkPoint:
        result = load('./checkpoint.pkl')
        evaluationsDone = len(result.x_iters)
        x0 = result.x_iters
        y0 = result.func_vals

        plotter = PlotterCallback(NSamples-evaluationsDone,tracker)

        result = gp_minimize(evaluate,            # the function to minimize
            [(0.0, 5.0),(0.0,5.0),(0.0,5.0)],    # the bounds on each dimension of x
            x0=x0,              # already examined values for x
            y0=y0,              # observed values for x0
            acq_func="LCB",     # the acquisition function (optional)
            n_calls=NSamples-evaluationsDone,         # number of evaluations of f including at x0
            n_initial_points=5,  # the number of random initialization points
            callback=[checkpoint_saver,plotter],
            random_state=777)

    else:    

        plotter = PlotterCallback(NSamples,tracker)
        result = gp_minimize(evaluate,            # the function to minimize
            [(0.0, 5.0),(0.0,5.0),(0.0,5.0)],    # the bounds on each dimension of x
            acq_func="LCB",     # the acquisition function (optional)
            n_calls=NSamples,         # number of evaluations of f including at x0
            #n_random_starts=3,  # the number of random initial points
            n_initial_points=5,  # the number of random initial points
            callback=[checkpoint_saver,plotter],# a list of callbacks including the checkpoint saver
            random_state=777)

    print(result)

    plt.subplots(figsize=(7, 7))
    plot_convergence(result);
    plt.pause(0.5)
    plt.show()



if __name__ == "__main__":
    #main("checkpoint_gen_9.pkl",True)
    main()









