import numpy as np
import matplotlib.pyplot as plt
from skopt import callbacks
from time import time



class PlotterCallback(callbacks.VerboseCallback):

    fx = None
    ax = None

    def __init__(self, n_total, tracker, n_init=0, n_random=0):
        self.n_init = n_init
        self.n_random = n_random
        self.n_total = n_total
        self.iter_no = 1
        self.tracker = tracker

        self._start_time = time()
        self._print_info(start=True)
        self._plot_info(start=True)


    def _plot_info(self, start=True, y=0.0):
        iter_no = self.iter_no
        if start:
            #Create graph
            self.f, self.ax = plt.subplots(figsize=(7, 7))
            plt.pause(0.5)

        if iter_no <= self.n_init:
            self.ax.plot(iter_no,y,'go')
            self.f.canvas.draw()
            plt.pause(0.5)
        

        elif self.n_init < iter_no <= (self.n_random + self.n_init):
            self.ax.plot(iter_no,y,'go')
            self.f.canvas.draw()
            plt.pause(0.5)

        else:
            self.ax.plot(iter_no,y,'go')
            self.f.canvas.draw()
            plt.pause(0.5)
            

    def __call__(self, res):
        """
        Parameters
        ----------
        res : `OptimizeResult`, scipy object
            The optimization as a OptimizeResult object.
        """
        
        curr_y = res.func_vals[-1]
        curr_min = res.fun

        self.tracker.step = self.iter_no

        time_taken = time() - self._start_time
        self._print_info(start = False)
        self._plot_info(start = False,y=curr_y)

        print("Time taken: %0.4f" % time_taken)
        print("Function value obtained: %0.4f" % curr_y)
        print("Current minimum: %0.4f" % curr_min)

        self.iter_no += 1
        if self.iter_no <= self.n_total:
            self._print_info(start=True)
            self._start_time = time()