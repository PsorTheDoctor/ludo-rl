import numpy as np
import matplotlib.pyplot as plt


def display_q_table(q_table):
    fig, axes = plt.subplots(1, 3, figsize=(18, 3), sharey=True)
    action_types = ['Safe actions', 'Unsafe actions', 'Home actions']
    actions = ['Move out', 'Move dice', 'Goal', 'Star', 'Globe', 'Protect', 'Kill', 'Die', 'Goal zone']
    n = len(actions)

    for i in range(3):
        table = q_table[:, i*n : i*n+n]
        ax = axes[i]
        ax.matshow(table, cmap='viridis')

        for j in range(table.shape[0]):
            for k in range(table.shape[1]):
                if table[j, k] == 0:
                    text = ax.text(k, j, '0', ha='center', va='center', color='w')
                else:
                    text = ax.text(k, j, f'{table[j, k]:.2f}', ha='center', va='center', color='w')

        ax.set_title(action_types[i])
        if i == 0:
            ax.set_ylabel('States')
        ax.set_xticks(np.arange(n), labels=actions)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(rotation=90)

    plt.tight_layout()
    plt.show()
