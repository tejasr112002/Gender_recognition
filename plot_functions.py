import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_histories(histories, folds=None, style="bmh"):
    """Plot the training and validation loss for each model in histories."""
    if "gender_loss" in histories[0].keys():
        no_subplots = 3
    else:
        no_subplots = 2
    if folds is None:
        folds = [i + 1 for i in range(len(histories))]
    plt.style.use(style)
    handles = []
    handles.append(Line2D([0], [0], color="k", linestyle="-", label="train"))
    handles.append(Line2D([0], [0], color="k", linestyle="--", label="val"))

    fig, ax = plt.subplots(1, no_subplots, figsize=(6.5 * no_subplots, 4.5))
    for j, history in enumerate(histories):
        if no_subplots == 3:
            metrics = ["loss", "gender_accuracy", "age_accuracy"]
            title = ["Cross-entropy loss", "gender accuracy", " age accuracy"]
        else:
            metrics = ["loss", "accuracy"]
            title = ["Cross-entropy loss", "Classification accuracy"]
        epochs = range(len(history["loss"]))
        for i, metric in enumerate(metrics):
            train = history[metric]
            val = history["val_" + metric]
            c = f"C{j+1}"
            ax[i].plot(epochs, train, "-", color=c)
            ax[i].plot(epochs, val, "--", color=c)
            ax[i].set_xlabel("epoch")
            ax[i].set_ylabel(metric)
            ax[i].set_title(title[i])
            # plot text with final value
            ax[i].text(epochs[-1], train[-1], f"{train[-1]:.3f}", color=c)
            ax[i].text(epochs[-1], val[-1], f"{val[-1]:.3f}", color=c)
        handles.append(
            Line2D([0], [0], color=c, linestyle="-", label=f"fold {folds[j]}")
        )

    x_max = len(histories[0]["loss"]) * 1.1
    for i in range(0, no_subplots):
        ax[i].legend(handles=handles, loc="upper right", fontsize=12)  # add legend
        ax[i].set_xlim(0, x_max)  # extend x-axis by 10%

    fig.tight_layout()
