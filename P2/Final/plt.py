import matplotlib.pyplot as plt
import json
import numpy as np

with open("stats.txt", "r") as f:
    n_lines = sum(1 for _ in f)
    f.seek(0)
    for line in range(n_lines):
        data = np.array((json.loads(f.readline()))["eval_loss_stats"])
        plt.plot(data, label=f"{line} : {min(data)}")
    plt.legend()
    plt.show()
    