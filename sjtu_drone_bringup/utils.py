import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)

    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('Running average of previous scores')
    directory = os.path.dirname(figure_file)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(figure_file)