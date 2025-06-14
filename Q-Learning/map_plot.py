import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Plot Map
def plot_frozenlake_map(env, filename="frozenlake_map.png"):
    desc = env.unwrapped.desc.astype('U1')
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


# Plot Q Table
def plot_q_table(env, q_table, map_name, is_slippery):
    import matplotlib.pyplot as plt
    import numpy as np

    actions = ['←', '↓', '→', '↑']
    desc = env.unwrapped.desc.astype(str)  # Mappa con lettere (S, F, G, H)
    grid_size = desc.shape[0]

    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    ax.set_xticks(np.arange(grid_size+1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size+1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rendi il background chiaro
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(grid_size-0.5, -0.5)  # inverti y

    for row in range(grid_size):
        for col in range(grid_size):
            state = row * grid_size + col
            cell = desc[row, col]

            # Colore dello sfondo in base al tipo di cella
            if cell == 'S':
                color = "#FF2B2B"  # azzurrino
            elif cell == 'F':
                color = "#D2E9FF"  # bianco
            elif cell == 'H':
                color = "#00B7FF"  # rosso chiaro
            elif cell == 'G':
                color = "#FFA500"  # verde chiaro
            else:
                color = "#dddddd"

            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=color))

            # Etichetta della cella
            ax.text(col, row - 0.35, cell, ha='center', va='center', fontsize=16, weight='bold')

            # Q-values in stile bussola
            for action in range(4):
                q_val = q_table[state, action]
                if action == 0: pos = (col - 0.2, row)      # LEFT
                elif action == 1: pos = (col, row + 0.2)    # DOWN
                elif action == 2: pos = (col + 0.2, row)    # RIGHT
                elif action == 3: pos = (col, row - 0.2)    # UP
                ax.text(pos[0], pos[1], f"{q_val:.2f}", ha='center', va='center', fontsize=7)

    plt.title("Q-table sovrapposta alla mappa", fontsize=14)
    filename = f"qtable_map_{map_name}_{'slippery' if is_slippery else 'deterministic'}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
