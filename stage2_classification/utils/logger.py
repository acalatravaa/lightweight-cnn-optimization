"""
ECE 565 - Computer Vision Final Project
Training Logger Utility

Records training metrics to a text file and provides visualization tools 
to plot the loss and accuracy curves during or after training.
"""

import os
import sys
from typing import List, Dict, Optional, Union

import matplotlib
# Force matplotlib to not use any Xwindows backend if running on a headless server
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Logger:
    """Save training process to log file with simple plot function."""
    
    def __init__(self, fpath: str, title: Optional[str] = None, resume: bool = False): 
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        self.names: List[str] = []
        self.nums_d: Dict[str, List[float]] = {}
        
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                
                for name in self.names:
                    self.nums_d[name] = []

                for numbers in self.file:
                    parsed_numbers = numbers.rstrip().split('\t')
                    for i in range(len(parsed_numbers)):
                        self.nums_d[self.names[i]].append(float(parsed_numbers[i]))
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names: List[str]) -> None:
        """Initializes the column headers for the log file."""
        if self.resume: 
            return
            
        self.names = names
        for name in self.names:
            self.nums_d[name] = []
            
        if self.file is not None:
            self.file.write('\t'.join(self.names) + '\n')
            self.file.flush()

    def append(self, numbers: List[Union[int, float]]) -> None:
        """Appends a new row of metrics to the log file."""
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        
        for index, num in enumerate(numbers):
            self.nums_d[self.names[index]].append(num)
            
        if self.file is not None:
            self.file.write('\t'.join([f'{x:.6f}' for x in numbers]) + '\n')
            self.file.flush()

    def plot(self, names: Optional[List[str]] = None) -> None:
        """Plots the specified metrics using Matplotlib."""
        names = self.names if names is None else names
        numbers = self.nums_d
        
        for name in names:
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
            
        legend_text = [f"{self.title}({name})" for name in names]
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)

    def close(self) -> None:
        """Safely closes the active log file."""
        if self.file is not None:
            self.file.close()


class LoggerMonitor:
    """Load and plot multiple loggers simultaneously."""
    
    def __init__(self, paths: Dict[str, str]):
        """
        Args:
            paths (Dict[str, str]): A dictionary of {name: filepath} pairs.
        """
        self.loggers: List[Logger] = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names: Optional[List[str]] = None) -> None:
        plt.figure()
        plt.subplot(121)
        legend_text = []
        
        for logger in self.loggers:
            plot_names = logger.names if names is None else names
            for name in plot_names:
                x = np.arange(len(logger.nums_d[name]))
                plt.plot(x, np.asarray(logger.nums_d[name]))
                legend_text.append(f"{logger.title}({name})")
                
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


def savefig(fname: str, dpi: Optional[int] = None) -> None:
    """Helper to save the current matplotlib figure."""
    dpi = 150 if dpi is None else dpi
    plt.savefig(fname, dpi=dpi)