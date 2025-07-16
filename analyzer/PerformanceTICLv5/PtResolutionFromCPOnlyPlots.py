import sys
sys.path.append("../..")
from functools import partial
from typing import Literal

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import math
import mplhep as hep
plt.style.use(hep.style.CMS)
hep.cms.label("Private work (CMS simulation)", loc=2)

import hist
import ROOT
from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *
from tqdm import tqdm
import os, ROOT
import cmsstyle as CMS
import multiprocessing
import pickle 

CMS.SetExtraText("Private work (CMS simulation)")
CMS.SetLumi("")

from numba import prange,njit

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path


fileV5 = "/eos/user/t/tipaulet/Local_Energy_Samples/SinglePionTiming_1p9_50GeV/histo/"
# fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/TimeResolution/SinglePionTiming_2p2_100GeV/histo/"

OutputDir = "/eos/user/t/tipaulet/www/PtResolutionCP1/"

create_directory(OutputDir)

pTlist= [5.000136144156897, 10.00006707134024, 20.000031764635192, 30.000021676344776, 40.00000874365381, 60.00001202543256, 79.99999974931147, 100.00000493045695, 150.00001241554511, 199.99996938137187]
sigmaOverE= [0.5549852920570127, 0.5016431839173484, 0.28878481356814667, 0.2200890201606922, 0.18808201600867833, 0.15315655847917567, 0.13675238909839085, 0.12283327358870301, 0.11231704633871172, 0.10089837367243262]
sigmaOverEErr=[0.0009928876005049867, 0.0005236276732968442, 0.0002628870225304075, 0.00016582161599103506, 0.0001286690540632052, 8.358742194087167e-05, 6.411381801707542e-05, 4.9847936778645295e-05, 3.273902775903686e-05, 2.5337108574374505e-05]
#(18.8114906563786, 37.62220924634356, 75.24403332644627, 112.86595228315986, 150.48786053868193, 225.73178670704846, 300.9756543435512, 376.21958765770705, 564.329400372255, 752.4390230234554)


#plot_ratio_single(numerator, denominator, 10, [1,200], label1="TICLv5", xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_v5.png")

def hyperbolic(x, A, B,C,D):
    return A / (x-B) + C*x + D

def PlotRes(tag="test",eta=1.9):
    


    with open('PtResolutionFromCP_'+str(eta)+'.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    x_data = data_dict['x_data']
    y_data = data_dict['y_data']
    y_errors = data_dict['y_errors']

    plt.clf()
    plt.errorbar(x_data,y_data,y_errors,label="Tracker $\sigma(p_T)/p_T$",color="black",marker="^",linestyle='',capsize=3,capthick=2)
    
    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)     
    if eta==1.9:
        plt.text(600, 0.344, "Single pion 0PU\n$\eta="+str(eta)+"$")
    if eta == 2.2:
        plt.text(600, 0.325, "Single pion 0PU\n$\eta="+str(eta)+"$")
    #plt.ylim(0,0.022)
    
    plt.xlabel("Regresssed $p_T$ TracksterCP [GeV]")
    #plt.ylabel("$\sigma(p_T)/p_T$")
    plt.ylabel("Resolution")

    # Usa curve_fit per fittare la funzione ai dati



    popt, pcov = curve_fit(hyperbolic, x_data, y_data, sigma=y_errors)

    # Estrai i parametri del fit
    A_fit, B_fit, C_fit , D_fit= popt

    # Crea i punti per la curva fittata
    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit = hyperbolic(x_fit, A_fit, B_fit, C_fit, D_fit)
    plt.errorbar(pTlist,sigmaOverE,sigmaOverEErr,label="HGCAL $\sigma(E)/E$ @ $|\eta|$=2.0",color="blue",marker="o",linestyle='',capsize=3,capthick=2)
    plt.plot(x_fit, y_fit, label=f'Fit: A={A_fit:.2g}, B={B_fit:.2g}, C={C_fit:.2g}, D={D_fit:.2g} ', color='red')
    






    plt.legend()
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"SigmaEdEvspT.png",bbox_inches='tight')


    plt.clf()
    plt.yscale("log")
    plt.xscale("log")
    plt.errorbar(x_data,y_data,y_errors,label="Tracker $\sigma(p_T)/p_T$",color="black",marker="^",linestyle='',capsize=3,capthick=2)
    plt.errorbar(pTlist,sigmaOverE,sigmaOverEErr,label="HGCAL $\sigma(E)/E$ @ $|\eta|$=2.0",color="blue",marker="o",linestyle='',capsize=3,capthick=2)

    plt.plot(x_fit, y_fit, label=f'Fit: A={A_fit:.2g}, B={B_fit:.2g}, C={C_fit:.2g}, D={D_fit:.2g} ', color='red')

    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    #plt.text(60, 0.014, "Single pion 0PU\n$\eta="+str(eta)+"$")
    #plt.ylim(0,0.022)
    
    plt.legend()
    plt.xlabel("Regresssed $p_T$ TracksterCP [GeV]")
    #plt.ylabel("$\sigma(p_T)/p_T$")
    plt.ylabel("Resolution")

    plt.savefig(OutputDir + tag+"eta"+str(eta)+"SigmaEdEvspTLOG.png",bbox_inches='tight')





#print(ReadFileAndFit(fileV5))


#folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV"]


if __name__ == "__main__":



    eta=1.9
    


    PlotRes(tag="pT",eta=eta)


    eta=2.2



    PlotRes(tag="pT",eta=eta)
