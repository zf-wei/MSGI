from auxpack.utils import generate_output
import matplotlib.pyplot as plt

#########################################################################
def Plot_Separate(output:bool, random_disturb:bool, MEAN, STD, filename):
    # Generate x-coordinates for the data points with a step size of 0.05
    x_values = np.arange(0.05, 0.86, 0.05)

    # Create a figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

    # Plot the lines for columns 1 in the first subplot
    ax1.plot(x_values, [MEAN[i][0] for i in range(len(MEAN))], label='Eucl & NMI', color='tab:red')
    ax1.fill_between(x_values, np.subtract([MEAN[i][0] for i in range(len(MEAN))], [STD[i][0] for i in range(len(STD))]),
                     np.add([MEAN[i][0] for i in range(len(MEAN))], [STD[i][0] for i in range(len(STD))]), alpha=0.2, color='tab:red')
    ax1.set_ylabel('NMI')
    ax1.legend()

    # Plot the lines for columns 2 in the second subplot
    ax2.plot(x_values, [MEAN[i][1] for i in range(len(MEAN))], label='Cosn & NMI', color='tab:green')
    ax2.fill_between(x_values, np.subtract([MEAN[i][1] for i in range(len(MEAN))], [STD[i][1] for i in range(len(STD))]),
                     np.add([MEAN[i][1] for i in range(len(MEAN))], [STD[i][1] for i in range(len(STD))]), alpha=0.2, color='tab:green')
    ax2.set_ylabel('NMI')
    ax2.legend()

    # Plot the lines for columns 3 in the third subplot
    ax3.plot(x_values, [MEAN[i][2] for i in range(len(MEAN))], label='Eucl & ECSim', color='tab:blue')
    ax3.fill_between(x_values, np.subtract([MEAN[i][2] for i in range(len(MEAN))], [STD[i][2] for i in range(len(STD))]),
                     np.add([MEAN[i][2] for i in range(len(MEAN))], [STD[i][2] for i in range(len(STD))]), alpha=0.2, color='tab:blue')
    ax3.set_ylabel('ECSim')
    ax3.legend()
    
    # Plot the lines for columns 4 in the fourth subplot
    ax4.plot(x_values, [MEAN[i][3] for i in range(len(MEAN))], label='Cosn & ECSim', color='tab:orange')
    ax4.fill_between(x_values, np.subtract([MEAN[i][3] for i in range(len(MEAN))], [STD[i][3] for i in range(len(STD))]),
                     np.add([MEAN[i][3] for i in range(len(MEAN))], [STD[i][3] for i in range(len(STD))]), alpha=0.2, color='tab:orange')
    ax4.set_xlabel('Percentage of Nodes Removed')
    ax4.set_ylabel('ECSim')

    # Remove the legend for "Column 4" in the fourth subplot
    #ax4.legend(handles=ax4.lines[:-1])

    # Set the x-axis scale
    plt.xticks(np.arange(0.0, 0.9, 0.05))

    # Automatically determine the lower bound for the y-axis
    y_min = min([min(y) for y in MEAN])
    y_max = max([max(y) for y in MEAN])
    y_range = y_max - y_min
    y_offset = y_range * 0.1  # Adjust the offset as needed
    y_lower = y_min - y_offset - 0.01
    y_upper = 1.02
    ax1.set_ylim(y_lower, y_upper)
    ax2.set_ylim(y_lower, y_upper)
    ax3.set_ylim(y_lower, y_upper)
    ax4.set_ylim(y_lower, y_upper)

    # Set the y-axis tick marks
    y_tick_step = 0.025
    y_ticks = np.arange(np.ceil(y_lower * 10) / 10, y_upper + 0.5*y_tick_step, y_tick_step)
    # Add horizontal reference lines
    for y in y_ticks:
        ax1.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        ax2.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        ax3.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        ax4.axhline(y=y, color='gray', linestyle='--', alpha=0.3)

    ax1.set_yticks(y_ticks)
    ax2.set_yticks(y_ticks)
    ax3.set_yticks(y_ticks)
    ax4.set_yticks(y_ticks)

    # Adjust the figure size
    fig.set_size_inches(10, 20)  # Increase the height of the figure
    fig.suptitle('Separate Plots',y=0.92)
    
    if output:
        filename = filename+"_SEPARATE"

        file_path = generate_output(random_disturb, filename+".png")
        plt.savefig(file_path)    

    # Show the plot
    plt.show()

#########################################################################