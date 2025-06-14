import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Plot Map
def plot_frozenlake_map(env, filename="frozenlake_map.png"):
    desc = env.unwrapped.desc.astype(str)
    nrow, ncol = desc.shape

    colors = {
        'F': "#D2E9FF",
        'H': "#00B7FF",
        'S': "#FF2B2B",
        'G': "#FFA500"
    }

    img = np.zeros((nrow, ncol, 3))
    for r in range(nrow):
        for c in range(ncol):
            tile = desc[r, c]
            img[r, c] = mcolors.to_rgb(colors.get(tile, '#FFFFFF'))

    fig, ax = plt.subplots(figsize=(ncol, nrow))
    ax.imshow(img)

    for r in range(nrow):
        for c in range(ncol):
            ax.text(c, r, desc[r, c], ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=2)
    ax.set_title(f"FrozenLake Map ({nrow}x{ncol})")

    plt.savefig(filename)
    plt.close()

# Plot Q Table
def plot_q_table(env, q_table, map_name, is_slippery):
    desc = env.unwrapped.desc.astype(str)
    n = desc.shape[0]

    fig, ax = plt.subplots(figsize=(n, n))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)

    colors = {
        'S': "#FF2B2B",
        'F': "#D2E9FF",
        'H': "#00B7FF",
        'G': "#FFA500"
    }

    for r in range(n):
        for c in range(n):
            state = r * n + c
            tile = desc[r, c]
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=colors[tile]))

            ax.text(c, r - 0.35, tile, ha='center', va='center', fontsize=16, fontweight='bold')

            for a in range(4):
                q = q_table[state, a]
                if a == 0: pos = (c - 0.2, r)    
                elif a == 1: pos = (c, r + 0.2)  
                elif a == 2: pos = (c + 0.2, r)  
                elif a == 3: pos = (c, r - 0.2)  
                ax.text(pos[0], pos[1], f"{q:.2f}", ha='center', va='center', fontsize=7)

    title = f"Q-table after training ({map_name}, {'slippery' if is_slippery else 'deterministic'})"
    ax.set_title(title, fontsize=12)

    filename = f"qtable_map_{map_name}_{'slippery' if is_slippery else 'deterministic'}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
