import numpy as np
import awkward as ak
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import mplhep as hep
import os
from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.energy_resolution.fit import *

plt.style.use(hep.style.CMS)


def clopper_pearson_interval(k, n, alpha=0.32):
    efficiency = k / n
    lower_bound = stats.beta.ppf(alpha / 2, k, n - k + 1)
    upper_bound = stats.beta.ppf(1 - alpha/2, k + 1, n - k)

    lower_error = efficiency - lower_bound
    upper_error = upper_bound - efficiency

    return efficiency, lower_error, upper_error


def compute_ratio(values1, values2, bins, rangeV=None):
    if (isinstance(values1, pd.core.series.Series)):
      values1 = values1.to_numpy()
    if (isinstance(values2, pd.core.series.Series)):
      values2 = values2.to_numpy()
    if (rangeV != None):
      hist2, bin_edges = np.histogram(values2, bins=bins, range=rangeV)
      hist1, _ = np.histogram(values1, bins=bin_edges, range=rangeV)
    else:
      hist2, bin_edges = np.histogram(values2, bins=bins)
      hist1, _ = np.histogram(values1, bins=bin_edges)

    ratio = np.zeros_like(hist1, dtype=float)
    ratio_err_low = np.zeros_like(hist1, dtype=float)
    ratio_err_high = np.zeros_like(hist1, dtype=float)

    for i in range(len(hist1)):
        n1 = hist1[i]
        n2 = hist2[i]
        if n2 > 0:
            eff, err_low, err_high = clopper_pearson_interval(n1, n2)
            ratio[i] = eff
            ratio_err_low[i] = err_low
            ratio_err_high[i] = err_high
        else:
            ratio[i] = 0
            ratio_err_low[i] = 0
            ratio_err_high[i] = 0
    ratio = np.nan_to_num(ratio, 0)
    ratio_err_high = np.nan_to_num(ratio_err_high, 0)
    ratio_err_low = np.nan_to_num(ratio_err_low, 0)

    return ratio, ratio_err_low, ratio_err_high, hist1, hist2, bin_edges


def plot_ratio_single(values1, values2, bins, rangeX, label1="Data", color1='blue', ylabel="Efficiency", xlabel="Variable", saveFileName="ratio.png"):
   # Upper pad: histograms

    # Compute ratio
    plt.figure(figsize=(10, 10))
    ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = compute_ratio(
        values1, values2, bins=bins, rangeV=rangeX)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Lower pad: ratio
    plt.errorbar(bin_centers, ratio, yerr=[ratio_err_low, ratio_err_high],
                 color=color1, label=label1, fmt="o",  markersize=5, linestyle='-')
    hep.cms.text("Simulation", loc=0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(0, 1.1)

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(saveFileName)


def plot_ratio_multiple(values1: list, values2: list, bins, rangeX, labels: list, colors: list, ratio_label="Data/MC", ratio_color='black', ylabel="Efficiency", xlabel="Variable", doRatio=True, saveFileName="ratio.png"):
    if (doRatio):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=1000,
              gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # Upper pad: histograms
        bin_centers = None
        ratios = []
        errorsLow = []
        errorsHigh = []
        for num, den, lab, col in zip(values1, values2, labels, colors):
            r, r_err_low, r_err_high, _, _, bin_edges = compute_ratio(
                num, den, bins=bins, rangeV=rangeX)
            if (not isinstance(bin_centers, tuple)):
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Store ratio
            ratios.append(r)
            errorsLow.append(r_err_low)
            errorsHigh.append(r_err_high)
            ax1.errorbar(bin_centers, r, yerr=[
                         r_err_low, r_err_high], color=col, label=lab, capsize=3, fmt="o", markersize=5, linestyle='-'),

        hep.cms.text("Simulation", ax=ax1)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(True)

        # Compute ratio
        if (len(ratios) == 0):
            ratio = np.zeros_like(len(bin_centers), dtype=float)
            ratio_err_low = np.zeros_like(len(bin_centers), dtype=float)
            ratio_err_high = np.zeros_like(len(bin_centers), dtype=float)
        else:
            ratio = np.nan_to_num(np.array(ratios[0]) / np.array(ratios[1]), 0)
            ratio_err_low = np.sqrt((np.array(errorsLow[0]) / np.array(ratios[0]))**2 + (
                np.array(errorsHigh[0]) / np.array(ratios[0]))**2)
            ratio_err_high = np.sqrt((np.array(errorsLow[0]) / np.array(ratios[1]))**2 + (
                np.array(errorsHigh[0]) / np.array(ratios[1]))**2)

        # Lower pad: ratio
        ax2.errorbar(bin_centers, ratio, yerr=[ratio_err_low, ratio_err_high], color=ratio_color,
                     label=ratio_label, capsize=3, fmt="o", markersize=5, linestyle='-')
        ax2.set_ylabel(ratio_label)
        ax2.set_ylim(0, 2.5)

        # Set labels
        ax1.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(saveFileName)
        plt.close()
        return fig, (ax1, ax2)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10), dpi=1000)

        # Upper pad: histograms
        bin_centers = None
        ratios = []
        errorsLow = []
        errorsHigh = []
        for num, den, lab, col in zip(values1, values2, labels, colors):
            r, r_err_low, r_err_high, _, _, bin_edges = compute_ratio(num, den, bins = bins, rangeV = rangeX)
            if(not isinstance(bin_centers, tuple)):
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Store ratio
            ratios.append(r)
            errorsLow.append(r_err_low)
            errorsHigh.append(r_err_high)
            ax1.errorbar(bin_centers, r, yerr=[r_err_low, r_err_high], color=col, label=lab, capsize = 3,fmt = "o", markersize = 5, linestyle = '-'),

        hep.cms.text("Simulation", ax=ax1)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(True)
        # Set labels
        ax1.set_ylabel(ylabel)       
        ax1.set_xlabel(xlabel)      
        ax1.grid(True)

        plt.tight_layout()
        plt.savefig(saveFileName)
        plt.close()
        return fig, ax1  

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

def process_directories_with_prefix(root_dir, prefix, histoDir):
    """
    Process all directories in the root directory that have a certain prefix.

    Args:
    - root_dir (str): Root directory containing directories with prefixes.
    - prefix (str): Prefix to filter directories.

    Returns:
    - results (list): List of results obtained from processing each directory.
    """
    results = []

    # List all directories in the root directory
    directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    # Filter directories based on prefix
    target_directories = [os.path.join(root_dir, d+f"/{histoDir}/") for d in directories if d.startswith(prefix)]
    # Process each directory
    print(target_directories)
    for directory in target_directories:
        reader = DumperInputManager(directory)
        # Do something with reader, for example:
        # results.append(reader.tracksters)
        results.append(reader)  # Append reader instance for further processing

    return results
