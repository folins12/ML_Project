# map_plot.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_frozenlake_map(env, filename="frozenlake_map.png"):
    desc = env.unwrapped.desc.astype('U1')  # decode bytes to unicode strings
    nrow, ncol = desc.shape

    color_map = {
        'F': "#D2E9FF",  # frozen
        'H': "#00B7FF",  # hole
        'S': "#FF2B2B",  # start
        'G': "#FFA500"   # goal
    }

    rgb_grid = np.zeros((nrow, ncol, 3))
    for r in range(nrow):
        for c in range(ncol):
            tile = desc[r, c]
            hex_color = color_map.get(tile, '#FFFFFF')
            rgb_grid[r, c] = mcolors.to_rgb(hex_color)

    fig, ax = plt.subplots(figsize=(ncol, nrow))
    ax.imshow(rgb_grid)

    for r in range(nrow):
        for c in range(ncol):
            ax.text(c, r, desc[r, c], ha='center', va='center', fontsize=16, fontweight='bold', color='black')

    ax.set_xticks(np.arange(-0.5, ncol, 1))
    ax.set_yticks(np.arange(-0.5, nrow, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linewidth=2)
    ax.set_title(f'FrozenLake Map ({nrow}x{ncol})')

    plt.savefig(filename)
    plt.close()
    print(f"Map plot saved as '{filename}'")
