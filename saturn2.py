import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation
from matplotlib.patches import Ellipse


class Main:
    def __init__(
        self,
        num_of_node: int = 100,
        basis_of_width=0.4,
        basis_of_height=0.4,
        basic_size: float = 0.25,
        w_weight: float = -0.065,
        x_lim=5,
        y_lim=1.5,
        smooth=0.05,
    ):
        plt.style.use("dark_background")

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect("equal")
        self.ax.set_frame_on(True)
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        self.ax.set_xlim([-x_lim - 1, x_lim + 1])
        self.ax.set_ylim([-y_lim - 1, y_lim + 1])

        (
            self.num_of_node,
            self.basis_of_width,
            self.basis_of_height,
            self.basic_size,
            self.w_weight,
            self.x_lim,
            self.y_lim,
            self.smooth,
        ) = (
            num_of_node,
            basis_of_width,
            basis_of_height,
            basic_size,
            w_weight,
            x_lim,
            y_lim,
            smooth,
        )

        X = np.random.uniform(
            -self.x_lim, self.x_lim, size=(self.num_of_node, 1)
        )  # (num_of_node X 1)
        Y = np.random.uniform(
            -self.y_lim, self.y_lim, size=(self.num_of_node, 1)
        )  # (num_of_node X 1)

        self.XY = np.concatenate([X, Y], axis=1)  # (num_of_node X 2)

        self.ellipses = []

        self.color_and_alpha = np.random.rand(
            self.num_of_node, 4  # (num_of_node X 4) R, G, B, alpha
        )

        for i, (x, y) in enumerate(self.XY):
            weight_coeff: float = np.exp(self.w_weight * (x ** 2))

            ellipse = self.ax.add_artist(
                Ellipse(
                    xy=[x, weight_coeff * y],
                    width=weight_coeff * self.basis_of_width * np.random.rand(),
                    height=weight_coeff * self.basis_of_height * np.random.rand(),
                    facecolor=self.color_and_alpha[i, 0:3],
                    alpha=(weight_coeff * self.color_and_alpha[i, 3]),
                    angle=np.random.rand() * 90,
                )
            )
            self.ellipses.append(ellipse)

    def __call__(self):
        def update(i):
            for i, ellipse in enumerate(self.ellipses):
                x, y = self.XY[i]
                x = x + self.smooth
                if x >= self.x_lim:
                    x = x - (2 * self.x_lim)
                self.XY[i] = np.array([x, y])
                weight_coeff: float = np.exp(self.w_weight * (x ** 2))
                ellipse.set_center([x, weight_coeff * y])
                ellipse.set_width(weight_coeff * self.basis_of_width * np.random.rand())
                ellipse.set_height(
                    weight_coeff * self.basis_of_height * np.random.rand()
                )
                ellipse.set_alpha(weight_coeff * self.color_and_alpha[i, 3])
                ellipse.set_angle(np.random.rand() * 90)

            return self.ellipses

        ani = FuncAnimation(self.fig, update, interval=40, blit=True, save_count=1000)

        # Writer = animation.writers["ffmpeg"]
        # writer = Writer(fps=20, metadata=dict(artist="Me"), bitrate=1800)
        # ani.save("g.mp4", writer=writer)
        plt.show()


if __name__ == "__main__":
    main = Main()
    main()
