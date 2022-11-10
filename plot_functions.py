import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_histories(histories, folds=None, style="seaborn-whitegrid"):
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

    fig, ax = plt.subplots(1, no_subplots, figsize=(6 * no_subplots, 5))
    for j, history in enumerate(histories):
        if no_subplots == 3:
            metrics = ["loss", "gender_accuracy", "age_accuracy"]
            title = ["Joint Cross-Entropy Loss", "Gender Accuracy", " Age Accuracy"]
        else:
            metrics = ["loss", "accuracy"]
            title = ["Cross-Entropy Loss", "Classification Accuracy"]
        epochs = range(len(history["loss"]))
        for i, metric in enumerate(metrics):
            train = history[metric]
            val = history["val_" + metric]
            c = f"C{j+1}"
            ax[i].plot(epochs, train, "-", color=c)
            ax[i].plot(epochs, val, "--", color=c)
            ax[i].set_xlabel("epoch", fontsize=12)
            ax[i].set_ylabel(metric, fontsize=12)
            ax[i].set_title(title[i], fontsize=14)
            # plot text with final value
            ax[i].text(epochs[-1], train[-1], f"{train[-1]:.3f}", color=c)
            ax[i].text(epochs[-1], val[-1], f"{val[-1]:.3f}", color=c)
        handles.append(
            Line2D([0], [0], color=c, linestyle="-", label=f"fold {folds[j]}")
        )

    x_max = len(histories[0]["loss"]) * 1.15
    for i in range(0, no_subplots):
        # ax[i].legend(handles=handles, loc=[0.5,0], fontsize=12)  # add legend
        ax[i].set_xlim(0, x_max)  # extend x-axis by 10%
    # plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
    #        bbox_transform = plt.gcf().transFigure)
    # fig.legend(handles=handles, loc=[-0.5,-0.2])
    fig.legend(handles=handles, loc=[0.1, 0], fontsize=12, ncol=7)  # add legend
    # fig.tight_layout()
