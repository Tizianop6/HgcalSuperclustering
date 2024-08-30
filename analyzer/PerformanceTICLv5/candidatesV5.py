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

from numba import prange

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

fileV5 = "/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo/"
# fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/TimeResolution/SinglePionTiming_2p2_100GeV/histo/"

OutputDir = "/eos/user/a/aperego/www/prova/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
], limitFileCount=10)

denominator = []
numerator = []
for i in prange(len(dumperInputV5.inputReaders)):
    dumper = dumperInputV5.inputReaders[i].ticlDumperReader
    tms = dumper.trackstersMerged
    cands = dumper.candidates
    simCands = dumper.simCandidates
    ass = dumper.associations
    sims = dumper.simTrackstersCP
    tracks = dumper.tracks
    print("file ", i)
    for ev in prange(len(tms)):
        tmEv = tms[ev]
        candEv = cands[ev]
        simCandEv = simCands[ev]
        assEv = ass[ev]
        simEv = sims[ev]
        tracksEv = tracks[ev]
        for simCand_idx in prange(len(simCandEv.simTICLCandidate_raw_energy)):
            simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
            simRegrEnergy = simCandEv.simTICLCandidate_regressed_energy[simCand_idx]
            simTrack = simCandEv.simTICLCandidate_track_in_candidate[simCand_idx]
            tk_idx_in_coll = -1
            try:
                tk_idx_in_coll = np.where(tracksEv.track_id == simTrack)[0][0] 
            except:
                continue
            if tk_idx_in_coll == -1 or tracksEv.track_pt[tk_idx_in_coll] < 1 or tracksEv.track_missing_outer_hits[tk_idx_in_coll] > 5 or not tracksEv.track_quality[tk_idx_in_coll]: 
                continue
                
            simToReco = assEv.ticlCandidate_simToReco_CP[simCand_idx]
            sharedE = assEv.ticlCandidate_simToReco_CP_sharedE[simCand_idx]
            score = assEv.ticlCandidate_simToReco_CP_score[simCand_idx]
            if not len(sharedE): continue
            if sharedE[ak.argmin(score)]/simRegrEnergy < 0.5: continue
            tid = simToReco[ak.argmin(score)]

            # obtain cand idx
            cand_idx = -1
            for i, k in enumerate(candEv.tracksters_in_candidate):
                if not len(k): continue
                if k[0] == tid:
                    cand_idx = i
                    break               
                # won't work with v4, use this instead:
                # for kk in k:
                #     if kk == simToReco_mergeTracksterCP[si][argminScore]:
                #         cand_idx = i
                #         break
            if cand_idx == -1: continue

            denominator.append(simRegrEnergy)

            recoTrack = candEv.track_in_candidate[cand_idx]
            if recoTrack == simTrack:
                # num += 1
                numerator.append(simRegrEnergy)

plt.figure()
plt.hist(denominator, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'blue', label="denominator")
plt.hist(numerator, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'red', label="numerator")
plt.legend()
plt.xlabel(("Sim Regressend Energy [GeV]"))
plt.savefig(OutputDir + "simTICLCand_regrEn.png")

plot_ratio_single(numerator, denominator, 10, [1,200], label1="TICLv5", xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_v5.png")




 