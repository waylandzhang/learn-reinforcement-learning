### File copied from https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


# Get the action with the max Q value
def get_argmax(G:np.ndarray) -> int:
    candidates = np.argwhere(G == G.max()).flatten()
    # return the only index if there's only one max
    if len(candidates) == 1:
        return candidates[0]
    else:
        # instead break the tie randomly
        return np.random.choice(candidates)


# Select arm and get the reward
def bandit(q_star:np.ndarray,
           act:int) -> tuple:
    real_rewards = np.random.normal(q_star, 1.0)
    # optim_choice = int(real_rewards[act] == real_rewards.max())
    optim_choice = int(q_star[act] == q_star.max())
    return real_rewards[act], optim_choice


# Plot results
def plot(data:np.ndarray,
         legends:list,
         xlabel:str,
         ylabel:str,
         filename:str=None,
         fn=lambda:None,) -> None:

    fontdict={
        'fontsize': 12,
        'fontweight': 'bold',
    }

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(c='lightgray')
    plt.margins(0.02)

    # revers the loop for a better visualization
    colors = ['cornflowerblue', 'tomato', 'lightseagreen']
    for i in range(len(data)-1, -1, -1):
        # data[i] = uniform_filter(data[i])
        plt.plot(data[i], label=legends[i], linewidth=1.5, c=colors[i])

    # get rid of the top/right frame lines
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)

    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel(ylabel, fontdict=fontdict)
    plt.legend(loc=4, fontsize=13)
    fn()

    if not filename:
        plt.show()
    else:
        plt.savefig(f'./plots/{filename}')