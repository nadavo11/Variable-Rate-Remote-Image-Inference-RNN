import matplotlib.pyplot as plt
from time import sleep
import random

# Initialize the plot outside the function to avoid re-creation
def init_loss_plot():
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'r-', label='Training Loss')  # Initialize an empty line
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    return ax,line

def plot_losses(losses,ax,line):
    ax.set_xlim(0, len(losses))  # Set the x-axis limits
    ax.set_ylim(min(losses), max(losses) + 0.1 * (max(losses) - min(losses)))  # Adjust the y-axis limits dynamically

    line.set_data(range(len(losses)), losses)  # Update the line data
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot

def train():
    ax, line = init_loss_plot()
    losses = []  # Initialize a list to store losses

    for epoch in range(3):  # Simulate 3 epochs for quicker demonstration
        for batch in range(50):  # Simulate 50 batches per epoch
            # Simulate computing loss
            loss = random.randrange(1,50)*(epoch + 1) * (batch + 1) * 0.01  # Example loss calculation
            losses.append(loss)

            if batch % 5 == 0:  # Update the plot more frequently for demonstration
                plot_losses(losses, ax, line)
    plt.plot(losses)
    plt.show()

# Execute the training function
train()