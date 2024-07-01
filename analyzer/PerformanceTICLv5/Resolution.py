import sys
sys.path.append("../..")
from functools import partial
from typing import Literal

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import hist

from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *

etaBinsText = ["1p9", "2p2","2p4", "2p5"]
inputPaths = ["/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/FixSkeletonsPU200/",
              "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/FixSkeletonsPU200/"
             ]

MergedListFits = [[],[]]
MergedListHistos = [[], []]
histoDirs = ['histo', 'histo']
labels = ['V5', 'V4']
dfs = []
for ih, (histoDir, label) in enumerate(zip(histoDirs, labels)):
    dfEta = []
    inputPath = inputPaths[ih]
    for etaText in etaBinsText:
        
        folderPrefix = "CloseByPionPU0PU_"+ etaText + "_150GeV"
        print(ih, etaText, folderPrefix)
        results = process_directories_with_prefix(inputPath, folderPrefix, histoDir)
        readers = []

        for i,r in enumerate(results):
            readers.extend(r.inputReaders)

        binsEdges = []

        pattern = rf"{etaText}_(\d+)"

        for res in readers:
            match = re.search(pattern, res.__repr__())
            binEnergy = float(match.group(1))
            binsEdges.append(binEnergy)

        histedges_equalN = np.unique(np.array(binsEdges))
        print(readers)
        res = runComputations([CPtoTracksterMerged_properties], readers, max_workers=24)
        df = res[0]
        dfEta.append(df)
        bin_edgesEnergy = [(i-0.5,i+0.5) for i in histedges_equalN ]
        it = 0
        histos = []
        histosCLUE3D = []
        for  minE, maxE in bin_edgesEnergy:
            histo = make_scOrTsOverCP_energy_histogram(name=f"tsOverCP_energy{label}",minEn = minE, maxEn = maxE, label=f"Trackster Merged - {label} / CaloParticle energy")
            fill_scOverCP_energy_histogram(histo, df, minE, maxE)
            histos.append(histo)


        histo_fit = fitMultiHistogram(histos)
        MergedListFits[ih].append(histo_fit)
        MergedListHistos[ih].append(histos)
    dfs.append(dfEta)

OutputDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5PerformancePostFix/Resolution/FixSkeletonsCloseByPion200PU/"


# Define the bin intervals
# bin_intervals = [(9, 11), (19, 21), (29, 31), (49, 51), (99, 101), (199, 201), (299, 301), (399, 401), (599, 601)]
bin_intervals = [(149,151)]
bin_intervals = bin_edgesEnergy

def categorize_energy(value):
    for interval in bin_intervals:
        if interval[0] <= value < interval[1]:
            return interval
    return None


##Single Plots
for i in range(len(labels)):
    oututDirResolutionRawEnergies = create_directory(OutputDir + f"/tracksterMerged/RawEnergies/{labels[i]}")
    oututDirResolution = create_directory(OutputDir + f"/tracksterMerged/Response/{labels[i]}")
    for ieta in range(len(dfs[i])):

        # Group by the bins
        df = dfs[i][ieta]
        df['energy_bin'] = df['regressed_energy_CP'].apply(categorize_energy)

        # Group by the new bin column
        grouped = df.groupby('energy_bin')
        for name, group in grouped:
            plt.figure(figsize = (10,10))
            plt.hist(group.raw_energy / group.raw_energy_CP, range = (0,2),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
            plt.savefig(f"{oututDirResolutionRawEnergies}/EnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
            plt.close()

            plt.figure(figsize = (10,10))
            plt.hist(group.raw_energy / group.raw_energy_CP, range = (0,2),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
            plt.savefig(f"{oututDirResolution}/Responses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
            plt.close()

##Single Plots
oututDirResolutionRawEnergies = create_directory(OutputDir + f"/tracksterMerged/RawEnergies/Comparison/")
oututDirResolution = create_directory(OutputDir + f"/tracksterMerged/Responses/Comparison")
for ieta in range(len(etaBinsText)):
  for i in range(len(labels)):
  
          # Group by the bins
          df = dfs[i][ieta]
          df['energy_bin'] = df['regressed_energy_CP'].apply(categorize_energy)
  
          # Group by the new bin column
          grouped = df.groupby('energy_bin')
          for name, group in grouped:
              plt.figure(figsize = (10,10))
              plt.hist(group.raw_energy, range = (0,name[1]),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
              plt.savefig(f"{oututDirResolutionRawEnergies}/EnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
              plt.close()
  
              plt.figure(figsize = (10,10))
              plt.hist(group.raw_energy, range = (0,name[1]),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
              plt.savefig(f"{oututDirResolution}/Responses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
              plt.close()
