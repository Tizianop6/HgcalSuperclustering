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
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties, CPtoTracksterAllShared_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

fileV5 =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByPionPU200PUTight/histo/"
fileV4 =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByPionPU200PUTight/histoV4/"

OutputDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/CloseByPion200PUTight/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
    ], 
    limitFileCount=None,
    )

dumperInputV4 = DumperInputManager([
    fileV4
    ], 
    limitFileCount=None,
    )


resV5 = runComputations([CPtoTrackster_properties, CPtoTracksterAllShared_properties], dumperInputV5, max_workers=10)
resV4 = runComputations([CPtoTrackster_properties, CPtoTracksterAllShared_properties], dumperInputV4, max_workers=10)

mergedV5 = resV5[0]
mergedV4 = resV4[0]

fig = plt.figure(figsize = (15,10))
energyBins = 100
etaBins = 50
phiBins = 50


#### BestRECO Plots #####
print(mergedV5)
outputDirTracksterMerged = create_directory(OutputDir + "/trackstersC3D/")
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy, range = (0,600),  bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy, range = (0,600),  bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(np.abs(mergedV5.barycenter_eta), range = (1.7, 2.7), bins = etaBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(np.abs(mergedV4.barycenter_eta), range = (1.7, 2.7), bins = etaBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy / mergedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy / mergedV4.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV4.raw_energy_CP / mergedV4.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTracksters", histtype = "step", lw = 2, color = 'black')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEnergyResponse.png")
plt.close()

### Efficient Reco Trackster Plots
# Filter the data based on the condition raw_energy/regressed_energy_CP >= 0.5
filtered_data_V5 = mergedV5[mergedV5['raw_energy'] / mergedV5['regressed_energy_CP'] >= 0.5]
filtered_data_V4 = mergedV4[mergedV4['raw_energy'] / mergedV4['regressed_energy_CP'] >= 0.5]

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.raw_energy, range = (0,600),  bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.raw_energy, range = (0,600),  bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(np.abs(filtered_data_V5.barycenter_eta), range = (1.7, 2.7), bins = etaBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(np.abs(filtered_data_V4.barycenter_eta), range = (1.7, 2.7), bins = etaBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.raw_energy / filtered_data_V4.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEnergyResponse.png")
plt.close()


##### SimTrackster Plots

outputDirSimTracksters = create_directory(OutputDir + "/simTracksters/")
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy_CP, range = (0,600),  bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy_CP, range = (0,600),  bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.regressed_energy_CP, range = (0,600),  bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.regressed_energy_CP, range = (0,600),  bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Regressed Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRegressedEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_eta_CP, range = (1.7, 2.7), bins = etaBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_eta_CP, range = (1.7, 2.7), bins = etaBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy_CP / mergedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "TICLv5", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy_CP / mergedV4.regressed_energy_CP, range = (0, 1.5),  bins = energyBins, label = "TICLv4", histtype = "step", lw = 2, color = 'blue')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEnergyResponse.png")
plt.close()

#### Responses ####
# Get the "viridis" colormap
import matplotlib as mpl

cmap = mpl.colormaps['Set1']

# Create a custom colormap with the desired number of colors
custom_cmap = ListedColormap(cmap(np.linspace(0, 1, 10)))

# Set the color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cmap.colors)

bins = [0, 20, 50, 100, 200 ,400]
data = [mergedV5, mergedV4]
lab = ['V5', 'V4']

outputDirTracksterMerged = create_directory(OutputDir + "/trackstersC3D/Responses/")
for id, d in enumerate(data):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Regressed")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponse" + lab[id] + ".png")
    plt.close()

for id, d in enumerate(data):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.raw_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Raw")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponseWrtRaw" + lab[id] + ".png")
    plt.close()


## Efficiencies ##
outputDirEffFakeMergeDup = create_directory(OutputDir + "/trackstersC3D/EffFakeMergeDup/")
filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= 0.5]
filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= 0.5]
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
plot_ratio_single(filtered_data_V5.raw_energy_CP, mergedV5.raw_energy_CP, energyBins, rangeX = (0,600), label1="TICLv5", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5.png")
plot_ratio_single(filtered_data_V4.raw_energy_CP, mergedV4.raw_energy_CP, energyBins, rangeX = (0,600), label1="TICLv4", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4.png")

nums = [filtered_data_V5.raw_energy_CP, filtered_data_V4.raw_energy_CP]
dens = [mergedV5.raw_energy_CP,  mergedV4.raw_energy_CP]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0,600), labels = ["TICLv5", "TICLv4"], colors = ['blue', 'red'], xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5TICLv4.png")

# Now we do the same thing for barycenter_eta and barycenter_phi
etaBins = 20
plot_ratio_single(filtered_data_V5.barycenter_eta_CP, mergedV5.barycenter_eta_CP, etaBins, rangeX = (1.7,2.7), label1="TICLv5", xlabel="Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5Eta.png")
plot_ratio_single(filtered_data_V4.barycenter_eta_CP, mergedV4.barycenter_eta_CP, etaBins, rangeX = (1.7,2.7), label1="TICLv4", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4Eta.png")

nums = [filtered_data_V5.barycenter_eta_CP, filtered_data_V4.barycenter_eta_CP]
dens = [mergedV5.barycenter_eta_CP,  mergedV4.barycenter_eta_CP]
plot_ratio_multiple(nums, dens, etaBins, rangeX = (1.7,2.7), labels = ["TICLv5", "TICLv4"], colors = ['blue', 'red'], xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5TICLv4Eta.png")

phiBins = 20
plot_ratio_single(filtered_data_V5.barycenter_phi_CP, mergedV5.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="TICLv5", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5Phi.png")
plot_ratio_single(filtered_data_V4.barycenter_phi_CP, mergedV4.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="TICLv4", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4Phi.png")

nums = [filtered_data_V5.barycenter_phi_CP, filtered_data_V4.barycenter_phi_CP]
dens = [mergedV5.barycenter_phi_CP,  mergedV4.barycenter_phi_CP]
plot_ratio_multiple(nums, dens, phiBins, rangeX = (-np.pi, np.pi), labels = ["TICLv5", "TICLv4"], colors = ['blue', 'red'], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5TICLv4Phi.png")


outputDirEffFakeMergeDup = create_directory(OutputDir + "/trackstersC3D/EffFakeMergeDupVarious/")
effTh = [0.5,0.6,0.7,0.8]
colors = ['blue', 'red', 'green', 'orange', 'purple']
mergedNum = []
mergedDen = []
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.raw_energy_CP)
    mergedDen.append(mergedV5.raw_energy_CP)
    labels.append(f"TICLv5 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,600), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_energy.png", doRatio = False)

mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.barycenter_eta_CP)
    mergedDen.append(mergedV5.barycenter_eta_CP)
    labels.append(f"TICLv5 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (1.7,2.7), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.barycenter_phi_CP)
    mergedDen.append(mergedV5.barycenter_phi_CP)
    labels.append(f"TICLv5 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_phi.png", doRatio = False)

#same thing for v4
mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.raw_energy_CP)
    mergedDen.append(mergedV4.raw_energy_CP)
    labels.append(f"TICLv4 - {th}")
print(len(mergedNum))
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,600), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4_energy.png", doRatio = False)


mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_eta_CP)
    mergedDen.append(mergedV4.barycenter_eta_CP)
    labels.append(f"TICLV4 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (1.7,2.7), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLV4_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_phi_CP)
    mergedDen.append(mergedV4.barycenter_phi_CP)
    labels.append(f"TICLV4 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLV4_phi.png", doRatio = False)

### Trackster sums

mergedV5TMP = resV5[1].reset_index()
mergedV4TMP = resV4[1].reset_index()

mergedV5 = mergedV5TMP.groupby(['eventInternal', 'caloparticle_id']).agg(
    raw_energy=('raw_energy', 'sum'),
    sharedE=('sharedE', 'sum'),
    raw_energy_CP=('raw_energy_CP', lambda x: list(x)[0]),
    regressed_energy_CP=('regressed_energy_CP', lambda x: list(x)[0]),
    barycenter_eta_CP=('barycenter_eta_CP', lambda x: list(x)[0]),
    barycenter_phi_CP=('barycenter_phi_CP', lambda x: list(x)[0])
).reset_index()


mergedV4 = mergedV4TMP.groupby(['eventInternal', 'caloparticle_id']).agg(
    raw_energy=('raw_energy', 'sum'),
    sharedE=('sharedE', 'sum'),
    raw_energy_CP=('raw_energy_CP', lambda x: list(x)[0]),
    regressed_energy_CP=('regressed_energy_CP', lambda x: list(x)[0]),
    barycenter_eta_CP=('barycenter_eta_CP', lambda x: list(x)[0]),
    barycenter_phi_CP=('barycenter_phi_CP', lambda x: list(x)[0])
).reset_index()

outputDirTracksterMerged = create_directory(OutputDir + "/sumTrackstersC3D/Responses/")
for id, d in enumerate(data):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Regressed")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponse" + lab[id] + ".png")
    plt.close()

for id, d in enumerate(data):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.raw_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Raw")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponseWrtRaw" + lab[id] + ".png")
    plt.close()


## Efficiencies ##
outputDirEffFakeMergeDup = create_directory(OutputDir + "/sumTrackstersC3D/EffFakeMergeDup/")
filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= 0.5]
filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= 0.5]
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
plot_ratio_single(filtered_data_V5.raw_energy_CP, mergedV5.raw_energy_CP, energyBins, rangeX = (0,600), label1="TICLv5", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5.png")
plot_ratio_single(filtered_data_V4.raw_energy_CP, mergedV4.raw_energy_CP, energyBins, rangeX = (0,600), label1="TICLv4", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4.png")

nums = [filtered_data_V5.raw_energy_CP, filtered_data_V4.raw_energy_CP]
dens = [mergedV5.raw_energy_CP,  mergedV4.raw_energy_CP]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0,600), labels = ["TICLv5", "TICLv4"], colors = ['blue', 'red'], xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5TICLv4.png")

# Now we do the same thing for barycenter_eta and barycenter_phi
etaBins = 20
plot_ratio_single(filtered_data_V5.barycenter_eta_CP, mergedV5.barycenter_eta_CP, etaBins, rangeX = (1.7,2.7), label1="TICLv5", xlabel="Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5Eta.png")
plot_ratio_single(filtered_data_V4.barycenter_eta_CP, mergedV4.barycenter_eta_CP, etaBins, rangeX = (1.7,2.7), label1="TICLv4", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4Eta.png")

nums = [filtered_data_V5.barycenter_eta_CP, filtered_data_V4.barycenter_eta_CP]
dens = [mergedV5.barycenter_eta_CP,  mergedV4.barycenter_eta_CP]
plot_ratio_multiple(nums, dens, etaBins, rangeX = (1.7,2.7), labels = ["TICLv5", "TICLv4"], colors = ['blue', 'red'], xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5TICLv4Eta.png")

phiBins = 20
plot_ratio_single(filtered_data_V5.barycenter_phi_CP, mergedV5.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="TICLv5", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5Phi.png")
plot_ratio_single(filtered_data_V4.barycenter_phi_CP, mergedV4.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="TICLv4", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4Phi.png")

nums = [filtered_data_V5.barycenter_phi_CP, filtered_data_V4.barycenter_phi_CP]
dens = [mergedV5.barycenter_phi_CP,  mergedV4.barycenter_phi_CP]
plot_ratio_multiple(nums, dens, phiBins, rangeX = (-np.pi, np.pi), labels = ["TICLv5", "TICLv4"], colors = ['blue', 'red'], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5TICLv4Phi.png")


outputDirEffFakeMergeDup = create_directory(OutputDir + "/sumTrackstersC3D/EffFakeMergeDupVarious/")
effTh = [0.5,0.6,0.7,0.8]
colors = ['blue', 'red', 'green', 'orange', 'purple']
mergedNum = []
mergedDen = []
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.raw_energy_CP)
    mergedDen.append(mergedV5.raw_energy_CP)
    labels.append(f"TICLv5 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,600), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_energy.png", doRatio = False)

mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.barycenter_eta_CP)
    mergedDen.append(mergedV5.barycenter_eta_CP)
    labels.append(f"TICLv5 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (1.7,2.7), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.barycenter_phi_CP)
    mergedDen.append(mergedV5.barycenter_phi_CP)
    labels.append(f"TICLv5 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_phi.png", doRatio = False)

#same thing for v4
mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.raw_energy_CP)
    mergedDen.append(mergedV4.raw_energy_CP)
    labels.append(f"TICLv4 - {th}")
print(len(mergedNum))
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,600), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4_energy.png", doRatio = False)


mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_eta_CP)
    mergedDen.append(mergedV4.barycenter_eta_CP)
    labels.append(f"TICLV4 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (1.7,2.7), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLV4_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_phi_CP)
    mergedDen.append(mergedV4.barycenter_phi_CP)
    labels.append(f"TICLV4 - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLV4_phi.png", doRatio = False)

