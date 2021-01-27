import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from multiprocessing import Queue
from typing import List
import numpy as np
from mpi4py import MPI
import time
import sys
import os
import signal
import pybullet as p
from pybullet_utils.bullet_client import BulletClient


CURVE_COLOR = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class DebugCurve:
    def __init__(self,
                 comm,
                 pause_time: float = 0.0,
                 num_curves: int = 0,
                 x_length: int = 0,
                 xlabel: str = None,
                 ylabel: str = None,
                 title: str = None,
                 xlim: List = None,
                 ylim: List = None,
                 grid: bool = False):
        self._comm = comm
        self._pause_time = pause_time
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title
        self._xlim = xlim
        self._ylim = ylim
        self._grid = grid

        self._num_lines = num_curves
        self._x_val_list = [0, ]
        self._y_val_list = [[0] for _ in range(self._num_lines)]

        self._x_length = x_length
        self._num_pOver = 0

        self._lines = []

        self.Reset()

    def Reset(self):
        # plt.cla()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(1, 1, 1)
        if self._xlabel is not None:
            self._ax.set_xlabel(self._xlabel)
        if self._ylabel is not None:
            self._ax.set_ylabel(self._ylabel)
        if self._title is not None:
            self._ax.set_title(self._title)

        for _ in range(self._num_lines):
            self._lines.append(self._ax.plot([0, 0], [0, 0])[0])
        plt.ion()
        if self._grid:
            plt.grid(linestyle='-.')

        if self._xlim is not None:
            self._ax.set_xlim(self._xlim)
        if self._ylim is not None:
            self._ax.set_ylim(self._ylim)

    def push_points(self):
        x_val = None
        y_val_list = []
        if isinstance(self._comm, type(Queue())):
            x_val, y_val_list = self._comm.get(True)
        elif isinstance(self._comm, MPI.Intracomm):
            x_val, y_val_list = self._comm.recv(source=0, tag=1)

        if len(self._x_val_list) >= self._x_length * 0.9:
            del self._x_val_list[0]
            for i in range(self._num_lines):
                del self._y_val_list[i][0]
            self._num_pOver += 1
            self._ax.set_xlim([i + self._num_pOver for i in self._xlim])

        self._x_val_list.append(x_val)
        for i in range(self._num_lines):
            self._y_val_list[i].append(y_val_list[i])

    def loop_func(self):
        n = 0
        while True:
            self.push_points()
            t1 = time.time()
            for i in range(self._num_lines):
                self._lines[i].set_xdata(np.asarray(self._x_val_list))
                self._lines[i].set_ydata(np.asarray(self._y_val_list[i]))
            plt.pause(self._pause_time)
            time_spent = time.time() - t1
            n += 1
            # if n % 100 == 0:
            #     print(f"time spent is: {time_spent * 1000} ms.")
