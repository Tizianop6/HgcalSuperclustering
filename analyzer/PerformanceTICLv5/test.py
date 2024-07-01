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

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

fileV5 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/FixSkeletonsPU200/CloseByPionPU0PU_2p5_150GeV/histo/"
fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByPionPU200PUNoPassThrough/histoV4/"

OutputDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/CloseByPion200PU/Test/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
], limitFileCount=None)

# dumperInputV4 = DumperInputManager([
#     fileV4
# ], limitFileCount=4)
            # Calculate ΔR and filter tracksters
def deltaR(eta1, phi1, eta2, phi2):
    dphi = phi2 - phi1
    while dphi > np.pi:
        dphi -= 2*np.pi
    while dphi < -np.pi:
        dphi += 2*np.pi
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)

puEnergy = []
sigEnergy = []
sigEnergyFirt5 = []
sigEnergyFirt10 = []
for i in range(len(dumperInputV5.inputReaders)):
    dumper = dumperInputV5.inputReaders[i].ticlDumperReader
    tm = dumper.trackstersMerged
    tr = dumper.tracksters
    ass = dumper.associations
    sim = dumper.simTrackstersCP
    for ev in range(len(tm)):
        tmEv = tm[ev]
        trEv = tr[ev]
        assEv = ass[ev]
        simEv = sim[ev]
        if(abs(simEv.barycenter_eta[0]) >= 0):
            simToReco = assEv.Mergetracksters_simToReco_CP[0]
            sharedE = assEv.Mergetracksters_simToReco_CP_sharedE[0]
            tid = simToReco[ak.argmax(sharedE)]
            simEnergy = simEv.raw_energy[0]
            simEta = simEv.barycenter_eta[0]
            simPhi = simEv.barycenter_phi[0]

            C3DsimToReco = assEv.tsCLUE3D_simToReco_CP[0]
            C3DsharedFraction = assEv.tsCLUE3D_simToReco_CP_sharedE[0] / simEnergy

            argsharedE = ak.sort(assEv.tsCLUE3D_simToReco_CP_sharedE[0])
            
            # print(argsharedE[-5:], argsharedE[:5])
            sharedTracksters = C3DsimToReco[np.where(C3DsharedFraction > 0.0)]
            puTracksters = C3DsimToReco[np.where(C3DsharedFraction == 0.0)]
            filteredPuTracksters = []
            for pt in puTracksters:
                trEta = trEv.barycenter_eta
                trPhi = trEv.barycenter_phi
                if(deltaR(simEta, simPhi, trEta[pt], trPhi[pt]) <= 1.0):
                    filteredPuTracksters.append(pt)

            # puEnergy.extend(trEv.raw_energy[filteredPuTracksters])
            # print(sharedTracksters)
            sigEnergy.extend([ak.sum(assEv.tsCLUE3D_simToReco_CP_sharedE[0]) / simEnergy ])
            sigEnergyFirt5.extend([ak.sum(argsharedE[-5:]) / simEnergy ])
            sigEnergyFirt10.extend([ak.sum(argsharedE[-10:]) / simEnergy ])

# plt.hist(puEnergy, bins = 100, label = "PU", histtype = "step", lw = 2, color = 'red')
plt.hist(sigEnergy, bins = 20, label = "C3D Tracksters sum", histtype = "step", lw = 2, color = 'blue')
plt.hist(sigEnergyFirt5, bins = 20, label = "C3D Tracksters sum best 5", histtype = "step", lw = 2, color = 'red')
plt.hist(sigEnergyFirt10, bins = 20, label = "C3D Tracksters sum best 10", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.xlabel(("Response wrt Sim Raw Energy"))
plt.savefig(OutputDir + "sigPUvsPU.png")




 