from os import stat
from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.animation import FuncAnimation


def _target_dist(x: float):
    answer = np.exp(-x)
    if (answer.ndim == 0) and (answer >= 1):
        answer = 0
    elif answer.ndim == 1:
        answer[answer >= 1] = 0

    return answer


class MetropolisHastingsAnimation:
    def __init__(self):
        self.fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
        for ax in axes:
            ax.spines.top.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)
            ax.set_yticks([])

        self.ax1, self.ax2 = axes
        self.ax1.set_xticks([])
        self.ax1.set_ylim([0, 1])
        self.ax2.set_xlim([0, 10])
        self.ax2.set_yticks([])

        self.X: List[float] = [1.5]
        self.x_space = np.linspace(-1, 5, 1000)

        current_X: float = self.X[-1]
        Q: Callable = stats.norm(current_X).pdf
        proposed_X = np.random.normal(loc=current_X, scale=1)

        self.ax1.plot(self.x_space, _target_dist(self.x_space))

        (self.Q_plot,) = self.ax1.plot(self.x_space, Q(self.x_space))
        (self.curr_x_line,) = self.ax1.plot(
            [current_X, current_X], [0, _target_dist(current_X)], linestyle=":"
        )
        (self.pro_x_line,) = self.ax1.plot(
            [proposed_X, proposed_X], [0, _target_dist(proposed_X)], linestyle=":"
        )

        self.text_pro_x = self.ax1.text(proposed_X, 0.1, r"$\hat x_1$", fontsize=18)
        self.text_curr_x = self.ax1.text(current_X, 0.1, "$x_0$", fontsize=18)

        acceptance_prob: float = np.minimum(
            1, _target_dist(proposed_X) / _target_dist(current_X)
        )
        RESULT = ""
        if np.random.uniform(0, 1) <= acceptance_prob:
            self.X.append(proposed_X)
            RESULT = "Accepted"
        else:
            self.X.append(current_X)
            RESULT = "Rejected"

        self.text_accept_prob = self.ax1.text(
            1.5, 0.6, f"Acceptance Prob: {acceptance_prob:.4f}\n{RESULT}", fontsize=20,
        )

        _, __, self.hist = self.ax2.hist(
            self.X, bins=np.linspace(0, 10, 30), density=True
        )

    def update(self, i):
        current_X = self.X[-1]
        Q = stats.norm(current_X).pdf
        proposed_X = np.random.normal(self.X[-1])
        # print(i, X)

        self.Q_plot.set_data(self.x_space, Q(self.x_space))
        self.curr_x_line.set_data([current_X, current_X], [0, _target_dist(current_X)])
        self.pro_x_line.set_data(
            [proposed_X, proposed_X], [0, _target_dist(proposed_X)]
        )

        acceptance_prob = np.minimum(
            1, _target_dist(proposed_X) / _target_dist(current_X)
        )
        RESULT = ""
        self.text_pro_x.set_x(proposed_X)
        self.text_pro_x.set_text(r"$\hat x_{" + str(i + 2) + "}$")
        self.text_curr_x.set_x(current_X)
        self.text_curr_x.set_text("$x_{" + str(i + 1) + "}$")

        if np.random.uniform(0, 1) <= acceptance_prob:
            self.X.append(proposed_X)
            RESULT = "Accepted"
        else:
            self.X.append(current_X)
            RESULT = "Rejected"

        self.text_accept_prob.set_text(
            f"Acceptance Prob: {acceptance_prob:.4f}\n{RESULT}"
        )

        self.ax2.clear()
        self.ax2.set_xlim([0, 10])
        self.ax2.set_yticks([])
        _, __, self.hist = self.ax2.hist(
            self.X, bins=np.linspace(0, 10, 30), density=True
        )

        return (
            self.Q_plot,
            self.curr_x_line,
            self.pro_x_line,
            self.text_accept_prob,
            self.text_curr_x,
            self.text_pro_x,
            self.ax2,
        )
        # return (ax1, ax2)

    def __call__(self):
        ani = FuncAnimation(fig=self.fig, func=self.update, blit=True, frames=100)
        plt.show()


if __name__ == "__main__":
    plt.style.use("dark_background")

    mh_anime = MetropolisHastingsAnimation()
    mh_anime()
