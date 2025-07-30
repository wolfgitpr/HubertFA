import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']


def plot_prob_to_image(melspec,
                       ph_seq,
                       ph_intervals,
                       frame_confidence,
                       cvnt_prob,
                       ph_time_gt=None,
                       label=None, v_min=-8, v_max=2, title=None,
                       bar_alpha=0.7, pcolor_alpha=0.4, frame_duration=None):
    label = label or [f'Tensor {i}' for i in range(len(cvnt_prob))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

    if title:
        fig.suptitle(title, fontsize=14)

    ph_seq = [i.split("/")[-1] for i in ph_seq]
    T = melspec.shape[-1]
    x = np.arange(T)

    ax1.imshow(melspec, origin="lower", aspect="auto", zorder=0)

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

    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=2, label='pred'),
        plt.Line2D([0], [0], color='b', lw=2, label='gt')
    ]
    ax1.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        ncol=1
    )

    melspec = melspec.T
    n_frames = melspec.shape[0]

    time_axis = np.arange(n_frames) * frame_duration
    time_edges = np.linspace(0, n_frames * frame_duration, n_frames + 1)

    scaled_prob = cvnt_prob * melspec.shape[1]
    prob_cumsum = np.cumsum(scaled_prob, axis=0)

    for i in range(len(scaled_prob)):
        if i == 0:
            ax2.bar(
                time_axis,
                scaled_prob[0],
                width=frame_duration if frame_duration else 1.0,
                align='edge' if frame_duration else 'center',
                alpha=0.0,
                label=label[0]
            )
        else:
            bottom = prob_cumsum[i - 1] if i > 0 else None
            ax2.bar(
                time_axis,
                scaled_prob[i],
                width=frame_duration if frame_duration else 1.0,
                align='edge' if frame_duration else 'center',
                bottom=bottom,
                label=label[i],
                alpha=bar_alpha
            )

    ax2.pcolormesh(
        time_edges,
        np.arange(melspec.shape[1] + 1),
        melspec.T,
        vmin=v_min,
        vmax=v_max,
        alpha=pcolor_alpha,
        shading='flat'
    )

    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Mel Bin')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Probability')

    if any(lbl is not None for lbl in label):
        ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    return fig
