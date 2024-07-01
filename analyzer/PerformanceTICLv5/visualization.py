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

fileV5 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByPionPU200PUNoPassThrough/histo/"
fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByPionPU200PUNoPassThrough/histoV4/"

OutputDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/CloseByPion200PU/Visualization/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
], limitFileCount=4)

# dumperInputV4 = DumperInputManager([
#     fileV4
# ], limitFileCount=4)


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
        if(simEv.barycenter_eta[0] >= 2.3):
            simToReco = assEv.Mergetracksters_simToReco_CP[0]
            sharedE = assEv.Mergetracksters_simToReco_CP_sharedE[0]
            tid = simToReco[ak.argmax(sharedE)]

            C3DsimToReco = assEv.tsCLUE3D_simToReco_CP[0]
            C3DsharedE = assEv.tsCLUE3D_simToReco_CP_sharedE[0]
            
            # Get the indices of the top 10 shared energies
            top_10_indices = np.argsort(C3DsharedE)[-10:]
            C3Dtid_top_10 = C3DsimToReco[top_10_indices]

            tm_vertices_x = tmEv.vertices_x[tid]
            tm_vertices_y = tmEv.vertices_y[tid]
            tm_vertices_z = tmEv.vertices_z[tid]
            sim_vertices_x = simEv.vertices_x[0]
            sim_vertices_y = simEv.vertices_y[0]
            sim_vertices_z = simEv.vertices_z[0]
            
            # Create the plots for the current event
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            sim_energy = simEv.raw_energy[0]
            reco_energy = tmEv.raw_energy[tid]
            shared_energy = sharedE[ak.argmax(sharedE)]

            sim_energy_formatted = f"{sim_energy:.1f}"
            reco_energy_formatted = f"{reco_energy:.1f}"
            shared_energy_formatted = f"{shared_energy:.1f}"

            axs[0, 0].scatter(sim_vertices_x, sim_vertices_y, color='blue', alpha=0.5, label=f"Simulated [E = {sim_energy_formatted}]")
            axs[0, 0].scatter(tm_vertices_x, tm_vertices_y, color='red', marker='x', alpha=0.5, label=f"Reconstructed [SE = {shared_energy_formatted} E = {reco_energy_formatted}]")
            
            # Plot for the top 10 C3Dtid
            for C3Dtid in C3Dtid_top_10:
                tr_vertices_x = trEv.vertices_x[C3Dtid]
                tr_vertices_y = trEv.vertices_y[C3Dtid]
                tr_vertices_z = trEv.vertices_z[C3Dtid]
                reco_energy = trEv.raw_energy[C3Dtid]
                shared_energy = C3DsharedE[C3Dtid]
                C3Dreco_energy_formatted = f"{reco_energy:.1f}"
                C3Dshared_energy_formatted = f"{shared_energy:.1f}"
                axs[0, 0].scatter(tr_vertices_x, tr_vertices_y, color='green', marker='X', alpha=0.5)

            axs[0, 0].set_xlabel('X')
            axs[0, 0].set_ylabel('Y')
            axs[0, 0].set_title('X-Y Projection')
            axs[0, 0].legend()

            # X-Z projection
            axs[0, 1].scatter(sim_vertices_z, sim_vertices_x, color='blue', alpha=0.5, label='Simulated')
            axs[0, 1].scatter(tm_vertices_z, tm_vertices_x, color='red', alpha=0.5, marker='x', label='Reconstructed')
            for C3Dtid in C3Dtid_top_10:
                tr_vertices_x = trEv.vertices_x[C3Dtid]
                tr_vertices_z = trEv.vertices_z[C3Dtid]
                axs[0, 1].scatter(tr_vertices_z, tr_vertices_x, color='green', alpha=0.5, marker='X')

            axs[0, 1].set_xlabel('Z')
            axs[0, 1].set_ylabel('X')
            axs[0, 1].set_title('X-Z Projection')
            axs[0, 1].set_xlim(300, 500)
            axs[0, 1].legend()

            # Y-Z projection
            axs[1, 0].scatter(sim_vertices_z, sim_vertices_y, color='blue', alpha=0.5, label='Simulated')
            axs[1, 0].scatter(tm_vertices_z, tm_vertices_y, color='red', marker='x', label='Reconstructed')
            for C3Dtid in C3Dtid_top_10:
                tr_vertices_y = trEv.vertices_y[C3Dtid]
                tr_vertices_z = trEv.vertices_z[C3Dtid]
                axs[1, 0].scatter(tr_vertices_z, tr_vertices_y, color='green', alpha=0.5, marker='X')

            axs[1, 0].set_xlabel('Z')
            axs[1, 0].set_ylabel('Y')
            axs[1, 0].set_title('Y-Z Projection')
            axs[0, 1].set_xlim(300, 500)
            axs[1, 0].legend()

            # R-Z projection
            sim_r = np.sqrt(sim_vertices_x**2 + sim_vertices_y**2)
            tm_r = np.sqrt(tm_vertices_x**2 + tm_vertices_y**2)
            axs[1, 1].scatter(sim_vertices_z, sim_r, color='blue', alpha=0.5, label='Simulated')
            axs[1, 1].scatter(tm_vertices_z, tm_r, color='red', alpha=0.5, marker='x', label='Reconstructed')
            for C3Dtid in C3Dtid_top_10:
                tr_vertices_x = trEv.vertices_x[C3Dtid]
                tr_vertices_y = trEv.vertices_y[C3Dtid]
                tr_vertices_z = trEv.vertices_z[C3Dtid]
                tr_r = np.sqrt(tr_vertices_x**2 + tr_vertices_y**2)
                axs[1, 1].scatter(tr_vertices_z, tr_r, color='green', alpha=0.5, marker='X')

            axs[1, 1].set_xlabel('Z')
            axs[1, 1].set_ylabel('R')
            axs[1, 1].set_title('R-Z Projection')
            axs[0, 1].set_xlim(300, 500)
            axs[1, 1].legend()

            # Save and show the plot
            plt.tight_layout()
            plot_filename = os.path.join(OutputDir, f'projections_event_{ev}.png')
            plt.savefig(plot_filename)
            plt.close(fig)
