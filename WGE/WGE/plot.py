#################################
#调用 generate_output 函数
#################################
import numpy as np
import matplotlib.pyplot as plt
from WGE.utils import generate_output

def Plot_Total(output:bool, random_disturb:bool, measurement:str, title:str, MEAN, STD, filename):
    # Generate x-coordinates for the data points with a step size of 0.05
    x_values = np.arange(0, 0.86, 0.05)

    # Create a figure and axes
    fig, ax = plt.subplots()

    labels = ['μ=0.01', 'μ=0.1', 'μ=0.2', 'μ=0.3', 'μ=0.4', 'μ=0.5']
    colors = ['red', 'orange', 'green', 'blue', 'indigo', 'violet']

    for i in range(len(labels)):
        ax.plot(x_values, [MEAN[j][i] for j in range(len(MEAN))], label=labels[i], color=colors[i])
        ax.fill_between(x_values, np.subtract([MEAN[j][i] for j in range(len(MEAN))], [STD[j][i] for j in range(len(STD))]),
                         np.add([MEAN[j][i] for j in range(len(MEAN))], [STD[j][i] for j in range(len(STD))]), alpha=0.2, color=colors[i])

    # Set labels and title
    ax.set_xlabel('Percentage of Nodes Removed')

    ax.set_ylabel(measurement)
    #ax.set_title('Total Plot')

    # Set the x-axis scale
    plt.xticks(np.arange(0.0, 0.9, 0.05))

    # Automatically determine the lower bound for the y-axis
    y_min = min([min(y) for y in MEAN])
    y_max = max([max(y) for y in MEAN])
    y_range = y_max - y_min
    y_offset = y_range * 0.1  # Adjust the offset as needed
    y_lower = 0 #y_min - y_offset - 0.01
    y_upper = 1.02
    ax.set_ylim(y_lower, y_upper)

    # Set the y-axis tick marks
    y_tick_step = 0.05
    y_ticks = np.arange(np.ceil(y_lower * 10) / 10, y_upper + 0.5*y_tick_step, y_tick_step)
    # Add horizontal reference lines
    for y in y_ticks:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        
    ax.set_yticks(y_ticks)

    # Add a legend
    ax.legend()

    # Adjust the figure size
    fig.set_size_inches(10, 6)
    fig.suptitle(title, y=0.95)

    if output:
        filename = filename

        file_path = generate_output(random_disturb, filename+".png")
        plt.savefig(file_path)    

    # Show the plot
    plt.show()