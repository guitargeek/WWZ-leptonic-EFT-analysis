import geeksw.plotting.cmsplot as plt
import numpy as np
from matplotlib import gridspec
import pandas as pd

# from wvz_helpers import sample_combinations


wvz_colors = {
    "ttz": "#f5ec45",
    "wz": "#e69f00",
    "higgs": "#009e73",
    "zz": "#5bbbf1",
    "twz": "#466dab",
    "othernoh": "#cccccc",
    "wwz": "r",
    "wzz": "b",
    "zzz": "#ffcc66",
    "nonh_wwz": "r",
    "nonh_wzz": "b",
    "nonh_zzz": "#ffcc66",
    "zh_wwz": "r",
    "wh_wzz": "b",
    "zh_zzz": "#ffcc66",
}

background_orders = {
    # "ttz": ["othernoh", "higgs", "wz", "zz", "twz", "ttz"],
    # "zz": ["othernoh", "twz", "higgs", "wz", "ttz", "zz"],
    "ttz": ["wz", "zz", "ttz"],
    "zz": ["wz", "ttz", "zz"],
}

legend_labels = {
    "ttz": "ttZ",
    "wz": "WZ",
    "higgs": "Higgs",
    "zz": "ZZ",
    "twz": "tWZ",
    "othernoh": "Other",
    "wwz": "WWZ",
    "wzz": "WZZ",
    "zzz": "ZZZ",
    "nonh_wwz": "NonH WWZ",
    "nonh_wzz": "NonH WZZ",
    "nonh_zzz": "NonH ZZZ",
    "zh_wwz": "ZH WWZ",
    "wh_wzz": "WH WZZ",
    "zh_zzz": "ZH ZZZ",
}


def make_ratioplot_axes():
    plt.figure(figsize=(8.57, 6.04 * (4.0 / 3)))

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    axis = plt.subplot(gs[0])
    axr = plt.subplot(gs[1])

    gs.update(wspace=0.025, hspace=0.075)
    plt.setp(axis.get_xticklabels(), visible=False)

    return axis, axr


def wvz_hist(
    data,
    column,
    bins,
    query=None,
    main_background="zz",
    plot_data=False,
    xlabel=None,
    ratio_plot=False,
    ylim=None,
    log_scale=False,
    legend_loc="auto",
):
    def select(df):
        if query is None or query == "":
            return df
        return df.query(query)

    # There is double counting on purpose here, such that vvv shows the sum of nonh_vvv and vh_vvv stacked
    # signal_components = ["wwz", "wzz", "zzz", "zh_wwz", "wh_wzz", "zh_zzz"]
    signal_components = ["wwz"]

    background_order = background_orders[main_background]

    if ratio_plot:
        axes = make_ratioplot_axes()
        plt.sca(axes[0])
    else:
        axes = None
        plt.figure()

    for label in signal_components:
        # if label in sample_combinations:
            # samples = sample_combinations[label]
        # else:
            # samples = [label]
        samples = [label]
        df = pd.concat([select(data[s]) for s in samples], ignore_index=True)

        dashed = "h_" in label

        plt.cms_hist(
            df[column],
            bins,
            weights=df["weight"],
            style="mc",
            label=legend_labels[label],
            color=wvz_colors[label],
            fill=False,
            dashed=dashed,
        )

    baseline_events = None
    baseline_errors2 = None

    for label in background_order:
        # if label in sample_combinations:
            # samples = sample_combinations[label]
        # else:
            # samples = [label]
        samples = [label]
        df = pd.concat([select(data[s]) for s in samples], ignore_index=True)

        plot_uncertainty = label == background_order[-1]
        baseline_events, baseline_errors2 = plt.cms_hist(
            df[column],
            bins,
            weights=df["weight"],
            style="mc",
            label=legend_labels[label],
            color=wvz_colors[label],
            fill=True,
            baseline_events=baseline_events,
            baseline_errors2=baseline_errors2,
            plot_uncertainty=plot_uncertainty,
        )

    if plot_data:
        data_events, data_errors2 = plt.cms_hist(
            select(data["data"])[column], bins, style="data", fill=False, color="r", label=r"Data"
        )

    if ylim:
        plt.ylim(ylim)

    plt.finalize(bins, log_scale=log_scale, xlabel=None if ratio_plot else xlabel, n_legend_cols=2)

    if not legend_loc == "auto":
        plt.legend(loc=legend_loc, ncol=2)

    if ratio_plot:
        plt.sca(axes[1])

        plt.xlim(bins[0], bins[-1])
        plt.ylim(0, 2)

        plt.xlabel(xlabel)
        plt.ylabel("Data/MC")

        plt.plot(plt.xlim(), [1, 1], color="k", linewidth=1)

        if plot_data:
            bin_centers = (bins[1:] + bins[:-1]) / 2.0
            y = data_events / baseline_events
            yerr = data_errors2 ** 0.5 / baseline_events
            yerr[y == 0] = np.nan
            y[y == 0] = np.nan
            plt.errorbar(bin_centers, y, yerr=yerr, color="k", fmt="o")

        baseline_errors = np.sqrt(baseline_errors2)
        x = np.vstack([bins, bins]).T.flatten()

        def to_y(events):
            return np.concatenate([[0.0], np.vstack([events, events]).T.flatten(), [0.0]])

        plt.fill_between(
            x,
            to_y((baseline_events - baseline_errors) / baseline_events),
            to_y((baseline_events + baseline_errors) / baseline_events),
            hatch="\\\\\\\\\\",
            facecolor="none",
            edgecolor="k",
            linewidth=0.0,
            alpha=0.5,
        )
