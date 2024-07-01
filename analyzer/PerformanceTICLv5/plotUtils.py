import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import mplhep as hep
from utils import *
import ROOT
plt.style.use(hep.style.CMS)


COLORS = ['red', 'blue', 'fuchsia']

class Plotter:
    def __init__(self, plot_dir, dataframe):
        self.plot_dir = plot_dir
        self.df = dataframe
        self._make_dir()

    def _make_dir(self):
        try:
            self.plot_dir = Path(self.plot_dir)
            if(not os.path.exists(self.plot_dir)):
              print(f"{bcolors.OKBLUE} Creating directory {self.plot_dir} {bcolors.ENDC}")
              self.plot_dir.mkdir(parents=True, exist_ok=True)
            else:
              print(f"{bcolors.OKBLUE} Directory {self.plot_dir} already exists {bcolors.ENDC}")
            plot_dir_str = str(self.plot_dir)  # Convert Path object to string
            if "eos" in str(plot_dir_str) and "www" in str(plot_dir_str):
                deploy_php(plot_dir_str)
        except Exception as e:
            print(f"Error creating directory {self.plot_dir}: {e}")

    def get_values(self, variable):
        try:
            values = self.df[variable].values
            if isinstance(values[0], (np.ndarray, list)):
                values = np.concatenate(values)
            return values
        except KeyError:
            print(f"Error: {variable} not found in DataFrame")
            return None
    def get_values(self, df, variable):
        try:
            values = df[variable].values
            if isinstance(values[0], (np.ndarray, list)):
                values = np.concatenate(values)
            return values
        except KeyError:
            print(f"Error: {variable} not found in DataFrame")
            return None


class SubPlot(Plotter):
    def __init__(self, plot_dir, dataframe, size=(15, 10), dpi=100, nrows=1, ncols=1):
        super().__init__(plot_dir, dataframe)
        self.size = size
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.figure, self.axes = plt.subplots(nrows, ncols, figsize=size, dpi=dpi)
        if nrows == 1 and ncols == 1:
            self.axes = np.array([self.axes])

    def hist(self, values, label="histo", histtype='step', color='fuchsia', bins=50, ret=False, density=False, range=(0, 0), ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        if range[0] == range[1]:
            range = None 

        ax.hist(values, histtype=histtype, color=color, bins=bins, label=label, density=density, range=range)
        hep.cms.text("Simulation", ax=ax)
        self.set_legend()
        if ret:
            return self.figure, ax

    def scatter(self, x, y, label="scatter", color='blue', marker='o', s=50, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.scatter(x, y, label=label, color=color, marker=marker, s=s)
        hep.cms.text("Simulation", ax=ax)

    def errorbar(self, x, y, yerr = None, label="scatter", color='blue', marker='o', s=8, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.errorbar(x, y, yerr=yerr, label=label, fmt = marker, color=color, ms=s, capsize = 4)
        hep.cms.text("Simulation", ax=ax)

    def line(self, x, y, label="line", color='green', linestyle='-', linewidth=2, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        hep.cms.text("Simulation", ax=ax)

    def bar(self, x, height, label="bar", color='purple', ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.bar(x, height, label=label, color=color)
        hep.cms.text("Simulation", ax=ax)

    def set_title(self, title, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.set_title(title)

    def set_xlabel(self, xlabel, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.set_xlabel(xlabel)

    def set_ylabel(self, ylabel, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.set_ylabel(ylabel)

    def set_xlim(self, xmin, xmax, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.set_xlim(xmin, xmax)

    def set_ylim(self, ymin, ymax, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.set_ylim(ymin, ymax)

    def set_legend(self, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.legend()

    def toggle_grid(self, grid=True, ax_idx=0):
        ax = self.axes.flatten()[ax_idx]
        ax.grid(grid)
        

class RatioPlot(SubPlot):
    def __init__(self, plot_dir, dataframe, size=(15, 10), dpi=100, nrows=1, ncols=1):
        super().__init__(plot_dir, dataframe)
        self.size = size
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.figure, self.axes = plt.subplots(nrows, ncols, figsize=size, dpi=dpi)
        if nrows == 1 and ncols == 1:
            self.axes = np.array([self.axes])

    def clopper_pearson_interval(self, k, n, alpha=0.32):
        efficiency = k / n
        lower_bound = stats.beta.ppf(alpha /2, k, n - k + 1)
        upper_bound = stats.beta.ppf(1 - alpha/2 , k + 1, n - k)
        
        lower_error = efficiency - lower_bound
        upper_error = upper_bound - efficiency
        
        return efficiency, lower_error, upper_error

    def compute_ratio(self, values1, values2, bins, rangeV = None):
        if(rangeV != None):
          hist2, bin_edges = np.histogram(values2, bins=bins, range = rangeV)
          hist1, _ = np.histogram(values1, bins=bin_edges, range = rangeV) 
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
                eff, err_low, err_high = self.clopper_pearson_interval(n1, n2)
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

    def plot_ratio(self, values1, values2, bins, label1="Data", label2="MC", ratio_label="Data/MC", color1='blue', color2='red', ratio_color='black'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.size, dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.figure = fig
        self.axes = np.array([ax1, ax2])

        # Upper pad: histograms
        h_val  = ax1.hist(values1, bins=bins, histtype='step', color=color1, label=label1)
        ax1.hist(values2, bins=bins, histtype='step', color=color2, label=label2)
        hep.cms.text("Simulation", ax=ax1)
        ax1.legend()

        # Compute ratio
        ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(values1, values2, bins = bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Lower pad: ratio
        ax2.errorbar(bin_centers, ratio, yerr=[ratio_err_low, ratio_err_high], color=ratio_color, label=ratio_label)
        hep.cms.text("Simulation", ax=ax2)
        ax2.set_ylabel(ratio_label)
        ax2.set_ylim(0, 2)

        # Set labels
        ax1.set_ylabel("Entries")
        ax2.set_xlabel("Variable")
        ax2.grid(True)

        plt.tight_layout()
        return self.figure, self.axes

    def plot_ratio_effs(self, values, errors, bins, labels, colors, xlabel = "", ylabel = "", ratio_label="Ratio", range = None):
        self.size = (12,12)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.size, dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.figure = fig
        self.axes = np.array([ax1, ax2])
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ref = values[-1]
        ref_err = errors[-1] 
        ratios = []
        ratios_err_low = []
        ratios_err_high = []
        for i_v, val in enumerate(values):
          ax1.errorbar(bin_centers, val, yerr=errors[i_v], fmt = "o", capsize = 2, color = colors[i_v], label = labels[i_v])
          ratio = val / ref 
          ratio_err_low = ratio * np.sqrt((np.array(ref_err[0]) / np.array(ref))**2 + (np.array(errors[i_v][0]) / np.array(val))**2)
          ratio_err_high = ratio * np.sqrt((np.array(ref_err[0]) / np.array(ref))**2 + (np.array(errors[i_v][0]) / np.array(val))**2)
          ratios.append(ratio)
          ratios_err_low.append(ratio_err_low)
          ratios_err_high.append(ratio_err_high)


        hep.cms.text("Simulation", ax=ax1)
        ax1.legend()
        ax1.set_ylim(0.,1.)
        if(range != None):
          ax1.set_xlim(range[0],range[1])
          ax2.set_xlim(range[0],range[1])

        for i_r, ratio in enumerate(ratios[:-1]):
          ax2.errorbar(bin_centers, ratio, yerr=[ratios_err_low[i_r], ratios_err_high[i_r]], color = colors[i_r], label = labels[i_r])

        ax2.set_ylabel(ratio_label)
#        ax2.set_ylim(-2, 2)

        # Set labels
        ax1.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.grid(True)

        plt.tight_layout()
        return self.figure, self.axes


    def save_plot(self, filename):
        saveName = os.path.join(self.plot_dir, f"{filename}.png") 
        plt.savefig(saveName)
        plt.close(self.figure)
        self.figure, self.axes = plt.subplots(self.nrows, self.ncols, figsize=self.size, dpi=self.dpi)
        if self.nrows == 1 and self.ncols == 1:
            self.axes = np.array([self.axes])

    def compute_efficiency(self, simTrackstersCP, trackstersMerged, associations, variable, label="", xLabel="PutLabelHere", bins=20, rangeV = None, eff_th=0.5, yLabel="Efficiency", prefixPlot="efficiency"):
        """
        Computes and plots the efficiency for given collections of simulated and merged tracksters.
        
        Parameters:
        - simTrackstersCP: Collection of simulated tracksters.
        - trackstersMerged: Collection of merged tracksters.
        - associations: Associations between simulated and merged tracksters.
        - variable: Variable to compute efficiency against.
        - label: Label for the plot.
        - xLabel: X-axis label.
        - bins: Number of bins for the histogram.
        - eff_th: Threshold for considering a match efficient.
        - yLabel: Y-axis label.
        - prefixPlot: Prefix for the saved plot file name.
        """
        if isinstance(simTrackstersCP, (list, np.array, np.ndarray)):
            self._compute_efficiency_multiple_collections(simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot)
        else:
            self._compute_efficiency_single_collection(simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot)


    def _compute_efficiency_single_collection(self, simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot):
        num = []
        den = []
        for ev, simTracksters in simTrackstersCP.iterrows():
            assoc = associations.iloc[ev, :]
            sharedE = assoc.Mergetracksters_simToReco_CP_sharedE
            for i in range(len(simTracksters.raw_energy)):
                var = getattr(simTracksters, variable)[i]
                den.append(var)
                if (sharedE[i] / simTracksters.raw_energy[i] >= eff_th).any():
                    num.append(var)
        ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(num, den, bins, rangeV)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.errorbar(bin_centers, ratio, yerr=[ratio_err_low, ratio_err_high], label=label)
        self.set_ylim(0, 1.)
        self.set_ylabel(yLabel)
        self.set_xlabel(xLabel)
        self.set_legend()
        self.save_plot(f"{prefixPlot}_{variable}")

    def _compute_efficiency_multiple_collections(self, simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot):
        assert len(simTrackstersCP) == len(trackstersMerged) == len(associations), "Collections have different dimensions"
        ratios = []
        errors = []
        for i_c, (simTrackstersTMP, trackstersMergedTMP, associationTMP) in enumerate(zip(simTrackstersCP, trackstersMerged, associations)):
            num = []
            den = []
            for ev, simTracksters in simTrackstersTMP.iterrows():
                assoc = associationTMP.iloc[ev, :]
                sharedE = assoc.Mergetracksters_simToReco_CP_sharedE
                for i in range(len(simTracksters.raw_energy)):
                    var = getattr(simTracksters, variable)[i]
                    den.append(var)
                    if (sharedE[i] / simTracksters.raw_energy[i] >= eff_th).any():
                        num.append(var)
            ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(num, den, bins, rangeV)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ratios.append(ratio)
            errors.append([ratio_err_low, ratio_err_high])
        self.plot_ratio_effs(ratios, errors, bin_edges, label, COLORS, xLabel, ylabel=yLabel)
        self.save_plot(f"{prefixPlot}_{variable}")

    def compute_fake_rate(self, simTrackstersCP, trackstersMerged, associations, variable, label="", xLabel="PutLabelHere", bins=20, rangeV = None, eff_th=0.6, yLabel="Fake Rate", prefixPlot="fake"):
        if isinstance(simTrackstersCP, (list, np.array, np.ndarray)):
            self._compute_fake_rate_multiple_collections(simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot)
        else:
            self._compute_fake_rate_single_collection(simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot)

    def _compute_fake_rate_single_collection(self, simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot):
        num = []
        den = []
        for ev, simTracksters in simTrackstersCP.iterrows():
            assoc = associations.iloc[ev, :]
            score = assoc.Mergetracksters_recoToSim_CP_score
            trackstersMergedEv = trackstersMerged.iloc[ev, :]
            for i in range(len(trackstersMergedEv.raw_energy)):
                var = getattr(trackstersMergedEv, variable)[i]
                den.append(var)
                if np.sum(score[i] <= eff_th) == 0:
                    num.append(var)
        ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(num, den, bins, rangeV)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.plot_ratio_effs(ratios, errors, bin_edges, label, COLORS, xLabel, ylabel=yLabel, range = (min(den), max(den)))
        self.save_plot(f"{prefixPlot}_{variable}")

    def _compute_fake_rate_multiple_collections(self, simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot):
        assert len(simTrackstersCP) == len(trackstersMerged) == len(associations), "Collections have different dimensions"
        ratios = []
        errors = []
        for i_c, (simTrackstersTMP, trackstersMergedTMP, associationTMP) in enumerate(zip(simTrackstersCP, trackstersMerged, associations)):
            num = []
            den = []
            for ev, simTracksters in simTrackstersTMP.iterrows():
                assoc = associationTMP.iloc[ev, :]
                score = assoc.Mergetracksters_recoToSim_CP_score
                trackstersMergedEv = trackstersMergedTMP.iloc[ev, :]
                for i in range(len(trackstersMergedEv.raw_energy)):
                    var = getattr(trackstersMergedEv, variable)[i]
                    den.append(var)
                    if np.sum(score[i] <= eff_th) == 0:
                        num.append(var)
            ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(num, den, bins, rangeV)
            ratios.append(ratio)
            errors.append([ratio_err_low, ratio_err_high])
        self.plot_ratio_effs(ratios, errors, bin_edges, label, COLORS, xLabel, ylabel=yLabel, range = (min(den), max(den)))
        self.save_plot(f"{prefixPlot}_{variable}")

    def compute_merge_rate(self, simTrackstersCP, trackstersMerged, associations, variable, label="", xLabel="PutLabelHere", bins=20, rangeV = None, eff_th=0.6, yLabel="Merge Rate", prefixPlot="merge"):
        if isinstance(simTrackstersCP, (list, np.array, np.ndarray)):
            self._compute_merge_rate_multiple_collections(simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot)
        else:
            self._compute_merge_rate_single_collection(simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot)

    def _compute_merge_rate_single_collection(self, simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot):
        num = []
        den = []
        for ev, simTracksters in simTrackstersCP.iterrows():
            assoc = associations.iloc[ev, :]
            score = assoc.Mergetracksters_recoToSim_CP_score
            trackstersMergedEv = trackstersMerged.iloc[ev, :]
            for i in range(len(trackstersMergedEv.raw_energy)):
                var = getattr(trackstersMergedEv, variable)[i]
                den.append(var)
                if np.sum(score[i] <= eff_th) > 1:
                    num.append(var)
        ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(num, den, bins, rangeV)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.plot_ratio_effs([ratio], [[ratio_err_low, ratio_err_high]], bin_edges, label, COLORS, xLabel, ylabel=yLabel)
        self.save_plot(f"{prefixPlot}_{variable}")
    def _compute_merge_rate_multiple_collections(self, simTrackstersCP, trackstersMerged, associations, variable, label, xLabel, bins, rangeV, eff_th, yLabel, prefixPlot):
        assert len(simTrackstersCP) == len(trackstersMerged) == len(associations), "Collections have different dimensions"
        ratios = []
        errors = []
        for i_c, (simTrackstersTMP, trackstersMergedTMP, associationTMP) in enumerate(zip(simTrackstersCP, trackstersMerged, associations)):
            num = []
            den = []
            for ev, simTracksters in simTrackstersTMP.iterrows():
                assoc = associationTMP.iloc[ev, :]
                score = assoc.Mergetracksters_recoToSim_CP_score
                trackstersMergedEv = trackstersMergedTMP.iloc[ev, :]
                for i in range(len(trackstersMergedEv.raw_energy)):
                    var = getattr(trackstersMergedEv, variable)[i]
                    den.append(var)
                    if np.sum(score[i] <= eff_th) > 1:
                        num.append(var)
            ratio, ratio_err_low, ratio_err_high, _, _, bin_edges = self.compute_ratio(num, den, bins, rangeV)
            ratios.append(ratio)
            errors.append([ratio_err_low, ratio_err_high])
        self.plot_ratio_effs(ratios, errors, bin_edges, label, COLORS, xLabel, ylabel=yLabel)
        self.save_plot(f"{prefixPlot}_{variable}")


def plot_trackster(collection, eosPlotDir, dataframes, directory, labels = None):
    # Select the dataframe based on the collection name
    collection_dir = eosPlotDir + directory + collection
    # Define bin configuration for each variable internally
    bin_config = {
        'raw_energy': 50,
        'regressed_energy': 50,
        'barycenter_eta': 15,
        'barycenter_phi': 15
    }

    xlabel_config = {
        'raw_energy': "Raw Energy [GeV]" ,
        'regressed_energy':  "Regressed Energy [GeV]",
        'barycenter_eta':  "eta",
        'barycenter_phi': "phi" 
    }
    # Create a directory for the collection if it doesn't exist
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)
    ratio_plot = RatioPlot(collection_dir, dataframes, size=(10, 8), dpi=100, nrows=1, ncols=1)
    if(isinstance(dataframes, dict)):
      dataframe = dataframes[collection]
      # Loop through each variable and create a separate plot
      for variable, bins in bin_config.items():
          # Create a RatioPlot instance for each variable
          values = ratio_plot.get_values(variable)
          # Plot histogram
          if(labels == None):
            label = variable
          ratio_plot.hist(values, label=label, bins=bins)
          # Set labels and titles
          ratio_plot.set_xlabel(variable)
          ratio_plot.set_ylabel("Entries")
          ratio_plot.set_xlabel(xlabel_config[variable])
          ratio_plot.save_plot(f'{collection}_{variable}_histogram')
    elif(isinstance(dataframes, (list, np.array, np.ndarray))):
      for variable, bins in bin_config.items():
          Range = None
          for idf, tmpdf in enumerate(dataframes):
              dataframe = tmpdf[collection]
              values = ratio_plot.get_values(dataframe, variable)
              if(Range == None):
                Range = (min(values), max(values))
              if(labels == None):
                label = variable
              else:
                label = labels[idf]
              ratio_plot.hist(values, label=label, color = COLORS[idf], bins=bins, range = Range)
          ratio_plot.set_xlabel(variable)
          ratio_plot.set_ylabel("Entries")
          ratio_plot.set_xlabel(xlabel_config[variable])
          ratio_plot.save_plot(f'{collection}_{variable}_histogram')
              

