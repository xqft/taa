import matplotlib.pyplot as plt

def config_plot():
    plt.grid(which="major", linestyle="-")
    plt.grid(which="minor", linestyle="-", alpha=0.1)
    plt.minorticks_on()