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
    T = melspec.shape[-1]
    x = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.imshow(melspec[0], origin="lower", aspect="auto", zorder=0)

    y_max = melspec.shape[-2]
    ax1.set_ylim(0, y_max)

    draw_upper_blue = ph_time_gt is not None
    red_upper, red_full = [], []

    for i, interval in enumerate(ph_intervals):
        if i == 0 or (i > 0 and ph_intervals[i - 1][1] != interval[0]):
            if 0 < interval[0] < T:
                if draw_upper_blue:
                    red_upper.append(interval[0])
                else:
                    red_full.append(interval[0])
        if 0 <= interval[1] < T:
            if draw_upper_blue:
                red_upper.append(interval[1])
            else:
                red_full.append(interval[1])

    if red_upper:
        ax1.vlines(red_upper, ymin=0.5 * y_max, ymax=y_max, colors='r', linewidth=1, zorder=2)
    if red_full:
        ax1.vlines(red_full, ymin=0, ymax=y_max, colors='r', linewidth=1, zorder=2)

    for i, interval in enumerate(ph_intervals):
        if ph_seq[i] == "SP":
            continue
        y_offset = len(ph_seq[i]) * T / 275
        x_center = (interval[0] + interval[1]) / 2
        ax1.text(
            x_center - y_offset,
            y_max + 1 if i % 2 else y_max - 6,
            ph_seq[i],
            fontsize=11,
            color="black" if i % 2 else "white",
            verticalalignment='bottom' if i % 2 else 'top',
            zorder=1
        )

    if draw_upper_blue:
        valid_times = [t for t in ph_time_gt if 0 <= t < T]
        ax1.vlines(valid_times, ymin=0, ymax=0.5 * y_max, colors='b', linewidth=1, zorder=2)

    y_scale = y_max
    ax1.plot(x, frame_confidence * y_scale, 'k-', lw=1, alpha=0.6, zorder=3)
    ax1.fill_between(x, frame_confidence * y_scale, color='k', alpha=0.3, zorder=3)

    ax1.legend(
        handles=[
            plt.Line2D([], [], color='r', lw=2, label='pred'),
            plt.Line2D([], [], color='b', lw=2, label='gt')
        ],
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
        extent=[0, x[-1], 0, ph_frame_prob.shape[1]]
    )
    ax2.plot(x, ph_frame_id_gt, 'r-', lw=1.5, zorder=2)
    edge_scale = ph_frame_prob.shape[-1]
    ax2.plot(x, edge_prob * edge_scale, 'k-', lw=1, zorder=2)
    ax2.fill_between(x, edge_prob * edge_scale, color='k', alpha=0.3, zorder=1)

    fig.set_size_inches(13, 7)
    plt.subplots_adjust(hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.05)
    return fig
