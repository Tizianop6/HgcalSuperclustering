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

OutputDir = "/eos/user/t/tipaulet/www/PtSigmaPulls/"

create_directory(OutputDir)

#@njit(parallel=True)
def ReadFileAndFit(filename,return_dict,key,tag="prova",eta=1.9, maxfiles=1):


    dumperInputV5 = DumperInputManager([
        filename
    ], limitFileCount=maxfiles)

    denominator = []
    numerator = []


    ticl_residuals={"avg":[],
                    "hgcal":[],
                    "mtd":[]}
    ptErr_list = []
    pulls_list = []

    #print("key",key)            
    for i in range(len(dumperInputV5.inputReaders)):
        #if(key==300):
        #    print("ciao")
        dumper = dumperInputV5.inputReaders[i].ticlDumperReader
        tms = dumper.trackstersMerged
        cands = dumper.candidates
        simCands = dumper.simCandidates
        ass = dumper.associations
        simtrackstersCP = dumper.simTrackstersCP
        simtrackstersSC = dumper.simTrackstersSC
        
        tracks = dumper.tracks
        #print("file ", i)

        for ev in range(len(tms)):
            tmEv = tms[ev]
            candEv = cands[ev]
            simCandEv = simCands[ev]
            assEv = ass[ev]
            tsCPEv = simtrackstersCP[ev]
            tracksEv = tracks[ev]
            tsSCEv = simtrackstersSC[ev]
            for simCand_idx in range(len(tsCPEv.barycenter_x)):
                #simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
                simRegrPt = tsCPEv.regressed_pt[simCand_idx]
                #if(key==300):
                #    print(simRegrPt)
                simTrackster = tsCPEv.trackIdx[simCand_idx]


                tk_idx_in_coll = -1
                try:
                    tk_idx_in_coll = np.where(tracksEv.track_id == simTrackster)[0][0] 
                except:
                    continue
                if tk_idx_in_coll == -1:
                    continue

                ptErr=tracksEv.track_ptErr[tk_idx_in_coll]
                pull = (tracksEv.track_pt[tk_idx_in_coll]-simRegrPt)/tracksEv.track_ptErr[tk_idx_in_coll]
                ptErr_list.append(ptErr)
                pulls_list.append(pull)
                
    #print(len(residual_list),key)

    plt.figure()
    #print(ticl_residuals)
    if (key>=50):
        rng=3.5
    elif (key< 6):
        rng= 0.06
    elif (key== 6):
        rng=0.2
    else:
        rng=0.5
    
    plt.clf()
    
    counts,bins,_=plt.hist(pulls_list, range=(-8,+8), bins = 100, histtype = "step", lw = 2, color = "xkcd:azure", label="Pulls")
    hep.cms.text("Private work (Simulation)", loc=0)                                                  
    
    plt.xlabel(("(Track $p_T$- SimTracksterCP $p_T$)/$p_T^{Error}$"))
    plt.ylabel("Entries")


    c1=ROOT.TCanvas("c1","c1",900,900)
    THistPulls = ROOT.TH1D("pT_"+str(key)+"GeV",";(Track p_{T}- SimTracksterCP p_{T})/p_{T}^{Error}; Entries", 100,-8.,8.)
    THistPulls.SetLineWidth(2)
    
    [THistPulls.Fill(x) for x in pulls_list]
    #THistPulls.Fill(np.array(pulls_list))
    ROOT.gStyle.SetOptFit(1111)
    gaussian = ROOT.TF1("gaussian", "gaus", -10, 10)

    fitres = THistPulls.Fit(gaussian,"L","",-4.,4.)
    THistPulls.Sumw2()
    #print(fitres.GetParams())
    sigma = gaussian.GetParameter(2)
    sigma_error = gaussian.GetParError(2)
    
    c1.SaveAs(OutputDir +"PullsRange4_" +tag+".png")
    """
    # Valori per l'asse x come i centri dei bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Fit della funzione gaussiana ai dati dell'istogramma
    popt, pcov = curve_fit(gauss, bin_centers, counts, sigma=np.sqrt(counts) ,absolute_sigma=True, p0=[1, 0, 1])

    # Estrazione dei parametri di fit
    A_fit, mu_fit, sigma_fit = popt
    sigma_errors = np.sqrt(np.diag(pcov))

    # Creazione della curva gaussiana
    x_fit = np.linspace(bins[0], bins[-1], 1000)
    y_fit = gauss(x_fit, *popt)

    # Plot dell'istogramma e della gaussiana fittata
    plt.plot(x_fit, y_fit, color='red', label=r'$\mu = {:.2f} \pm {:.2f}$,\n$\sigma = {:.2f} \pm {:.2f}$'.format(mu_fit, sigma_errors[1], sigma_fit, sigma_errors[2]))


    plt.legend()
    

    plt.savefig(OutputDir +"Pulls_" +tag+".png",bbox_inches='tight')
    """
    """
    plt.clf()
    

    y_h,x_h,_=plt.hist(ptErr_list, range=(0.,+rng), bins = 100, histtype = "step", lw = 2, color = "xkcd:magenta", label="Combination")
    #plt.hist(ticl_residuals["mtd"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = "xkcd:azure", label="ETL")
    #plt.hist(ticl_residuals["hgcal"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = 'xkcd:green', label="HGCAL")

    xlist=np.linspace(0,rng,1000)

    hlist=[]
    hlist.append(hist.Hist(hist.axis.Regular(100,0,rng,name="tres"), name="name", label="label"))
    hlist[0].fill(ptErr_list)


    #print(fitMultiHistogram(hlist))
    fitres=fitMultiHistogram(hlist)
    #print(fitres)
    #print(fitres)

    cruj=partial(cruijff, A=fitres[0][0].params.A, m=fitres[0][0].params.m, sigmaL=fitres[0][0].params.sigmaL,
        sigmaR=fitres[0][0].params.sigmaR, alphaL=fitres[0][0].params.alphaL, alphaR=fitres[0][0].params.alphaR)

    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color = "xkcd:azure")
    #print(cruj(np.array(xlist)))



    #print(fitres[0][0].params.A,fitres[0][1].params.A,fitres[0][2].params.A)
    plt.legend()
    plt.xlabel(("$p_T^{Error}$"))
    plt.ylabel("Entries")
    hep.cms.text("Private work (Simulation)", loc=0)                                                  
    
    match = re.search(r'(\d+)GeV', tag)
    
    plt.text(0.045, y_h.max()*0.4, "$p_T="+match.group(1)+"$ GeV\n$\eta="+str(eta)+"$")

    #plt.savefig(OutputDir + tag+".png",bbox_inches='tight')

    
    res=fitres[0][0].params.m
    cov_matrix=fitres[0][0].covMatrix
    resErr=(fitres[0][0].params.sigmaL+fitres[0][0].params.sigmaR)/2

    #points_list=np.array(residual_list)
    """
    resAndErrs=[sigma,sigma_error]


    return_dict[key]=resAndErrs
    
    #return (resAndErrs)

#plot_ratio_single(numerator, denominator, 10, [1,200], label1="TICLv5", xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_v5.png")


def PlotRes(dict_resolutions,tag="test",eta=1.9):
    #print(dict_resolutions)
    #print(dict_resolutions[2])
    counter=0
    pts={'x':[],'xpt':[],"y":[],"yErr":[],"y2":[],"y2Err":[]}
    for enkey in dict_resolutions.keys():
        #pts["x"].append(enkey*math.cosh(eta))
        pts["x"].append(enkey)
        pts["xpt"].append(enkey)
        pts["y"].append(dict_resolutions[enkey][0])
        pts["yErr"].append(dict_resolutions[enkey][1])
        pts["y2"].append(dict_resolutions[enkey][0]/(enkey))
        pts["y2Err"].append(dict_resolutions[enkey][1]/(enkey))




    """
    plt.clf()
    plt.errorbar(pts["x"],pts["y"],pts["yErr"],label="Response sigma",color="xkcd:magenta",marker="^",linestyle='',capsize=3,capthick=2)
    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 0PU\n$\eta="+str(eta)+"$")
    plt.ylim(0., 0.017)
    plt.legend()
    plt.xlabel("$Energy$ [GeV]")
    plt.ylabel("Response sigma")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"Sigma.png")


    plt.clf()
    
    plt.errorbar(pts["x"],pts["y2"],pts["y2Err"],label="Resolution",color="xkcd:magenta",marker="^",linestyle='',capsize=3,capthick=2)
    
    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 0PU\n$\eta="+str(eta)+"$")
    plt.legend()

    plt.xlabel("$Energy$ [GeV]")
    plt.ylabel("Resolution [1/GeV]")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"SigmaEdE.png")


    plt.clf()
    
    plt.errorbar(pts["xpt"],pts["y2"],pts["y2Err"],label="Resolution",color="xkcd:magenta",marker="^",linestyle='',capsize=3,capthick=2)
    
    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.014, "Single pion 0PU\n$\eta="+str(eta)+"$")
    plt.ylim(0,0.022)
    plt.legend()
    plt.xlabel("Regressed SimTracksterCP $p_T$ [GeV]")
    plt.ylabel("$\sigma(p_T)/p_T$")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"SigmaEdEvspT_RANGE.png",bbox_inches='tight')

    """


    canvasfit=ROOT.TCanvas(str(eta),str(eta),800,800)

    graph = ROOT.TGraphErrors(len(pts["xpt"]))#, np.array(pts["xpt"]), np.array(pts["y"]),None, np.array(pts["yErr"]))
    for i in range(len(pts["xpt"])):
        graph.SetPoint(i, pts["xpt"][i], pts["y"][i])
        graph.SetPointError(i, 0, pts["yErr"][i])

    graph.SetTitle("#eta="+str(eta)+";Sim p_{T}; #sigma(Pulls)")
    hyperbolic = ROOT.TF1("hyperbolic", "[0]/(x - [1]) + [2]", 0.1, 10)
    ROOT.gStyle.SetOptFit(1111)
    hyperbolic.SetParName(0, "a")   # Primo parametro: a
    hyperbolic.SetParName(1, "x0")  # Secondo parametro: x0
    hyperbolic.SetParName(2, "b")   # Terzo parametro: b
    hyperbolic.SetParameters(0.20, 2, 0.1)

    
    

    graph.SetLineColor(ROOT.kBlue)
    graph.SetMarkerColor(ROOT.kBlue)
    graph.SetMarkerStyle(34)
    graph.Fit(hyperbolic)
    a = hyperbolic.GetParameter(0)
    a_error = hyperbolic.GetParError(0)
    x0 = hyperbolic.GetParameter(1)
    x0_error = hyperbolic.GetParError(1)
    b = hyperbolic.GetParameter(2)
    b_error = hyperbolic.GetParError(2)
    parameters = {
    "a": a,
    "x0": x0,
    "b": b
    }

    # Dump su file .pkl
    with open("params_hyperbole"+str(eta)+".pkl", "wb") as f:
        pickle.dump(parameters, f)

    graph.Draw("AP")

    canvasfit.SaveAs(OutputDir+tag+"eta"+str(eta)+"fit.png")










#print(ReadFileAndFit(fileV5))


#folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV"]


if __name__ == "__main__":


    folders_1p9 = ["SinglePionTiming_1p9_1000GeV", "SinglePionTiming_1p9_500GeV", "SinglePionTiming_1p9_300GeV","SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV",
     "SinglePionTiming_1p9_15GeV", "SinglePionTiming_1p9_2GeV",
     "SinglePionTiming_1p9_30GeV", "SinglePionTiming_1p9_4GeV",
     "SinglePionTiming_1p9_50GeV", "SinglePionTiming_1p9_6GeV",
     "SinglePionTiming_1p9_8GeV"] 


    dict_resolutions={}

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = [] 

    eta=1.9
    maxfiles=50
    

    for i in tqdm(range(len(folders_1p9))):
        file_name="/eos/user/t/tipaulet/Samples/PtError/TimeResolutionPtError0PU/"+folders_1p9[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_1p9[i])

        p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_1p9[i],eta,maxfiles))
        jobs.append(p)
        p.start()

        
    for proc in jobs:
        proc.join()
    
    print(return_dict)

    dict_resolutions=return_dict

    PlotRes(dict_resolutions,tag="pT",eta=eta)


    

    folders_2p2 = ["SinglePionTiming_2p2_100GeV", "SinglePionTiming_2p2_10GeV",
     "SinglePionTiming_2p2_15GeV", "SinglePionTiming_2p2_2GeV",
     "SinglePionTiming_2p2_30GeV", "SinglePionTiming_2p2_4GeV",
     "SinglePionTiming_2p2_50GeV", "SinglePionTiming_2p2_6GeV",
     "SinglePionTiming_2p2_8GeV"]


    
    dict_resolutions={}

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = [] 

    eta=2.2
    maxfiles=50
    

    for i in tqdm(range(len(folders_2p2))):
        file_name="/eos/user/t/tipaulet/Samples/PtError/TimeResolutionPtError0PU/"+folders_2p2[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_2p2[i])

        p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        jobs.append(p)
        p.start()

        
    for proc in jobs:
        proc.join()
    
    print(return_dict)

    dict_resolutions=return_dict

    PlotRes(dict_resolutions,tag="pT",eta=eta)


    
