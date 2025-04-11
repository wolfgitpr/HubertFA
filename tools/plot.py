import matplotlib.pyplot as plt
import numpy as np


def plot_for_valid(
        melspec,
        ph_seq,
        ph_intervals,
        frame_confidence,
        ph_frame_prob,
        ph_frame_id_gt,
        edge_prob,
        ph_time_gt=None
):
    ph_seq = [i.split("/")[-1] for i in ph_seq]
    x = np.arange(melspec.shape[-1])

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(melspec[0], origin="lower", aspect="auto")

    draw_upper_blue = ph_time_gt is not None

    for i, interval in enumerate(ph_intervals):
        if i == 0 or (i > 0 and ph_intervals[i - 1][1] != interval[0]):
            if interval[0] > 0:
                if draw_upper_blue:
                    ax1.axvline(interval[0], color="r", linewidth=1, ymin=0.5, ymax=1.0)
                else:
                    ax1.axvline(interval[0], color="r", linewidth=1)

        if interval[1] < melspec.shape[-1]:
            if draw_upper_blue:
                ax1.axvline(interval[1], color="r", linewidth=1, ymin=0.5, ymax=1.0)
            else:
                ax1.axvline(interval[1], color="r", linewidth=1)
        if ph_seq[i] != "SP":
            if i % 2:
                ax1.text(
                    (interval[0] + interval[1]) / 2
                    - len(ph_seq[i]) * melspec.shape[-1] / 275,
                    melspec.shape[-2] + 1,
                    ph_seq[i],
                    fontsize=11,
                    color="black",
                )
            else:
                ax1.text(
                    (interval[0] + interval[1]) / 2
                    - len(ph_seq[i]) * melspec.shape[-1] / 275,
                    melspec.shape[-2] - 6,
                    ph_seq[i],
                    fontsize=11,
                    color="white",
                )

    if draw_upper_blue:
        for time in ph_time_gt:
            if 0 <= time < melspec.shape[-1]:
                ax1.axvline(time, color="b", linewidth=1, ymax=0.5)

    ax1.plot(
        x, frame_confidence * melspec.shape[-2], color="black", linewidth=1, alpha=0.6, label='_nolegend_'
    )
    ax1.fill_between(x, frame_confidence * melspec.shape[-2], color="black", alpha=0.3, label='_nolegend_')

    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=2, label='pred'),
        plt.Line2D([0], [0], color='b', lw=2, label='gt')
    ]

    ax1.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        ncol=2
    )

    ax2.imshow(
        ph_frame_prob.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        # vmin=0,
        # vmax=1,
    )

    ax2.plot(x, ph_frame_id_gt, color="red", linewidth=1.5)
    # ax2.scatter(x, ph_frame_id_gt, s=5, marker='s', color="red")

    ax2.plot(x, edge_prob * ph_frame_prob.shape[-1], color="black", linewidth=1)
    ax2.fill_between(x, edge_prob * ph_frame_prob.shape[-1], color="black", alpha=0.3)

    fig.set_size_inches(13, 7)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    return fig
