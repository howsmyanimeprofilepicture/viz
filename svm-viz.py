import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def main(use_legend=True, title="", use_textbox=True, use_frame=True, C=0, lr=0.005):
    # Create the SVM model
    model = torch.nn.Linear(2, 1)

    num_of_data = 30
    # create datapoints
    datapoints = torch.rand(num_of_data, 2)
    indices = []
    for feat_1, feat_2 in datapoints:
        if feat_2 + feat_1 <= 1:
            indices.append(True)
        else:
            indices.append(False)
    indices_inv = [b == False for b in indices]

    neg_data = datapoints[indices]
    pos_data = datapoints[indices_inv]
    y_true = [1 if idx else -1 for idx in indices]
    y_true = torch.tensor(y_true)

    # Definit functions
    def train(model, br_point=0.0005, epoch=100, lr=0.005, disp=False, C=C):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for i in range(epoch):
            optimizer.zero_grad()
            y_pred = model(datapoints)
            y_pred = y_pred.squeeze()
            loss = torch.sum(torch.clamp(-(y_true * y_pred) + 1, min=0))

            w_vec = model.weight.squeeze()
            penalty = (
                C * (w_vec.t() @ w_vec) / 2
            )  # The penalty about the narrow margin width
            # Becuase the margin width equals (norm(w)^2)/2
            loss += penalty

            if disp and i % 150 == 149:
                print("Loss: ", loss.item())
            if loss.item() <= br_point:
                break
            loss.backward()
            optimizer.step()

        return loss.item()

    # Visualization
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 15))
    if not use_frame:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_frame_on(use_frame)

    ax.scatter(
        neg_data[:, 0], neg_data[:, 1], label="Negative Data Points", color="orange"
    )
    ax.scatter(
        pos_data[:, 0], pos_data[:, 1], label="Positive Data Points", color="white"
    )

    (w_vec_plot,) = ax.plot([0], [0], color="skyblue")
    w_vec_plot.set_label("w vector")
    (db_line,) = ax.plot([0], [0], color="violet")
    db_line.set_label("Decision Boundary")

    (gutter1,) = ax.plot([0], [0], color="violet", linewidth=1, linestyle=":")
    (gutter2,) = ax.plot([0], [0], color="violet", linewidth=1, linestyle=":")
    text_box = ax.text(
        0.01, 0.01, "", fontsize=14, bbox={"color": "black", "alpha": 0.5,}
    )
    if use_legend:
        ax.legend(fontsize=14, loc="upper right")

    def update(i):
        loss = train(model, br_point=0.0005, epoch=100, lr=lr)

        w_1, w_2 = model.weight.squeeze().detach().numpy()
        b = model.bias.item()
        x_1 = np.linspace(0, 1, 10)
        x_2 = (-b - w_1 * x_1) / w_2
        db_line.set_xdata(x_1)
        db_line.set_ydata(x_2)

        x_2 = (1 - b - w_1 * x_1) / w_2
        gutter1.set_xdata(x_1)
        gutter1.set_ydata(x_2)

        x_2 = (-1 - b - w_1 * x_1) / w_2
        gutter2.set_xdata(x_1)
        gutter2.set_ydata(x_2)

        w_vec_plot.set_xdata([100 * w_1, -100 * w_1])
        w_vec_plot.set_ydata([100 * w_2, -100 * w_2])
        if use_textbox:
            text_box.set_text(
                f"""Loss: {loss.__round__(4)}
W Vector : [{float(w_1).__round__(2)}, {float(w_2).__round__(2)}]
Bias : {model.bias.item().__round__(2)}
Learning Rate : {lr.__round__(5)}
Iteration : {i}
C : {C}"""
            )

        return (db_line, gutter1, gutter2, w_vec_plot, text_box)
        # The animation function must return a sequence of Artist objects.

    ani = FuncAnimation(fig, update, interval=40, blit=True,)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", "-f", default="True")
    parser.add_argument("--textbox", "-tb", default="True")
    parser.add_argument("--legend", "-l", default="True")
    parser.add_argument("--learning_rate", "-lr", default="0.005")

    parser.add_argument("--title", "-t", default="SVM")
    parser.add_argument("--C", default="0.")
    args = parser.parse_args()
    print(eval(args.C))

    main(
        use_legend=eval(args.legend),
        title=args.title,
        use_frame=eval(args.frame),
        use_textbox=eval(args.textbox),
        C=eval(args.C),
        lr=eval(args.learning_rate),
    )

