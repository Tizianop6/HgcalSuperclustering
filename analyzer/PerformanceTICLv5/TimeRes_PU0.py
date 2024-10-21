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
import time

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

OutputDir = "/eos/user/t/tipaulet/www/TimeResolutionWFitInfo/"

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

    for i in prange(len(dumperInputV5.inputReaders)):
        dumper = dumperInputV5.inputReaders[i].ticlDumperReader
        tms = dumper.trackstersMerged
        cands = dumper.candidates
        simCands = dumper.simCandidates
        ass = dumper.associations
        sims = dumper.simTrackstersCP
        #tracks = dumper.tracks
        #print("file ", i)
        for ev in range(len(tms)):
            tmEv = tms[ev]
            candEv = cands[ev]
            simCandEv = simCands[ev]
            assEv = ass[ev]
            simEv = sims[ev]
            #tracksEv = tracks[ev]
            #print(tracksEv)
            for simCand_idx in range(len(simCandEv.simTICLCandidate_raw_energy)):
                simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
                simRegrEnergy = simCandEv.simTICLCandidate_regressed_energy[simCand_idx]
                simTime = simCandEv.simTICLCandidate_time[simCand_idx]
                simTrack = simCandEv.simTICLCandidate_track_in_candidate[simCand_idx]
                if simTrack == -1: 
                    continue
                '''
                tk_idx_in_coll = -1
                try:
                    tk_idx_in_coll = np.where(tracksEv.track_id == simTrack)[0][0] 
                except:
                    continue
                
                if tk_idx_in_coll == -1 or tracksEv.track_pt[tk_idx_in_coll] < 1 or tracksEv.track_missing_outer_hits[tk_idx_in_coll] > 5 or not tracksEv.track_quality[tk_idx_in_coll]: 
                    continue
                ''' 

                #start = time.process_time()
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
                #print("selection:",time.process_time() - start)


                denominator.append(simRegrEnergy)

                candidate_time = candEv.candidate_time[cand_idx]
                candidate_timeErr= candEv.candidate_timeErr[cand_idx]
                track_in_cand=candEv.track_in_candidate[cand_idx]
                #print(track_in_cand)
                if track_in_cand==-1:
                    continue
                candidate_time_MTD = candEv.candidate_time_MTD[cand_idx]
                candidate_time_MTDErr= candEv.candidate_time_MTD_err[cand_idx]


                t_HGCal = False
                t_MTD = False

                if (candidate_timeErr>0):
                    t_HGCal=True 
                    ticl_residuals["hgcal"].append(simTime-candidate_time)
                    #ticl_energies["hgcal"].append(candidate_energy[i][j])
                    #ticl_pt["hgcal"].append(math.sqrt(candidate_px[i][j]**2+candidate_py[i][j]**2))
                    #ticl_isCharged["hgcal"].append(not candidate_charge[i][j]==0)
                    time_avg=candidate_time
                    time_avgErr=candidate_timeErr

                if(candidate_time_MTDErr>0):
                    t_MTD=True
                    MTD_time=candidate_time_MTD
                    ticl_residuals["mtd"].append(simTime-MTD_time)
                    #ticl_energies["mtd"].append(candidate_energy[i][j])
                    #ticl_pt["mtd"].append(math.sqrt(candidate_px[i][j]**2+candidate_py[i][j]**2))
                    #ticl_isCharged["mtd"].append(not candidate_charge[i][j]==0)
                    if(t_HGCal):
                        inv_hgcal_err=1./(time_avgErr**2)
                        inv_mtd_err=1./(candidate_time_MTDErr**2)
                        time_avg=(candidate_time*inv_hgcal_err+MTD_time*inv_mtd_err)/(inv_mtd_err+inv_hgcal_err)
                    else:
                        time_avg=MTD_time

                if(t_MTD or t_HGCal):
                    ticl_residuals["avg"].append(simTime-time_avg)
                    #ticl_energies["avg"].append(candidate_energy[i][j])
                    #ticl_pt["avg"].append(math.sqrt(candidate_px[i][j]**2+candidate_py[i][j]**2))
                    #ticl_isCharged["avg"].append(not candidate_charge[i][j]==0)

                #if recoTrack == simTrack:
                    # num += 1
                #    numerator.append(simRegrEnergy)

    plt.figure()
    resAndErrs={"avg":[],"mtd":[],"hgcal":[]}

    #print(ticl_residuals)
    y_h,x_h,_=plt.hist(ticl_residuals["avg"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = "xkcd:magenta", label="Combination")
    plt.hist(ticl_residuals["hgcal"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = 'xkcd:green', label="HGCAL")

    xlist=np.linspace(-0.1,0.1,1000)

    hlist=[]
    hlist.append(hist.Hist(hist.axis.Regular(100,-0.1,0.1,name="tres"), name="name", label="label"))
    hlist[0].fill(ticl_residuals["avg"])

    hlist.append(hist.Hist(hist.axis.Regular(100,-0.1,0.1,name="tres"), name="name", label="label"))
    hlist[1].fill(ticl_residuals["mtd"])

    hlist.append(hist.Hist(hist.axis.Regular(100,-0.1,0.1,name="tres"), name="name", label="label"))
    hlist[2].fill(ticl_residuals["hgcal"])


    #print(fitMultiHistogram(hlist))
    fitres=fitMultiHistogram(hlist)
    #print(fitres)

    res=(fitres[0][1].params.sigmaL+fitres[0][1].params.sigmaR)/2
    cov_matrix=fitres[0][1].covMatrix
    resErr=math.sqrt(0.25*(cov_matrix[2][2]+cov_matrix[3][3])+ 0.5*cov_matrix[2][3])
    resAndErrs["mtd"]=[res,resErr]


    plt.hist(ticl_residuals["mtd"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = "xkcd:azure", 
        label=f"ETL \n$\sigma_L=${fitres[0][1].params.sigmaL:.3}$\pm${math.sqrt(cov_matrix[2][2]):.2} \n$\sigma_R=${fitres[0][1].params.sigmaR:.3}$\pm${math.sqrt(cov_matrix[3][3]):.2}")
    

    cruj=partial(cruijff, A=fitres[0][0].params.A, m=fitres[0][0].params.m, sigmaL=fitres[0][0].params.sigmaL,
        sigmaR=fitres[0][0].params.sigmaR, alphaL=fitres[0][0].params.alphaL, alphaR=fitres[0][0].params.alphaR)

    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color = "xkcd:magenta")
    #print(cruj(np.array(xlist)))


    cruj=partial(cruijff, A=fitres[0][1].params.A, m=fitres[0][1].params.m, sigmaL=fitres[0][1].params.sigmaL,
        sigmaR=fitres[0][1].params.sigmaR, alphaL=fitres[0][1].params.alphaL, alphaR=fitres[0][1].params.alphaR)

    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color = "xkcd:azure")


    cruj=partial(cruijff, A=fitres[0][2].params.A, m=fitres[0][2].params.m, sigmaL=fitres[0][2].params.sigmaL,
        sigmaR=fitres[0][2].params.sigmaR, alphaL=fitres[0][2].params.alphaL, alphaR=fitres[0][2].params.alphaR)

    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color = 'xkcd:green')


    #print(fitres[0][0].params.A,fitres[0][1].params.A,fitres[0][2].params.A)
    plt.legend()
    plt.xlabel(("Residual [ns]"))
    plt.ylabel("Entries")
    hep.cms.text("Private work (Simulation)", loc=0)                                                  
    
    match = re.search(r'(\d+)GeV', tag)
    
    plt.text(0.045, y_h.max()*0.4, "$p_T="+match.group(1)+"$ GeV\n$\eta="+str(eta)+"$")

    plt.savefig(OutputDir + tag+".png")

    
    res=(fitres[0][0].params.sigmaL+fitres[0][0].params.sigmaR)/2
    cov_matrix=fitres[0][0].covMatrix
    resErr=math.sqrt(0.25*(cov_matrix[2][2]+cov_matrix[3][3])+ 0.5*cov_matrix[2][3])
    resAndErrs["avg"]=[res,resErr]

    
    



    res=(fitres[0][2].params.sigmaL+fitres[0][2].params.sigmaR)/2
    cov_matrix=fitres[0][2].covMatrix
    resErr=math.sqrt(0.25*(cov_matrix[2][2]+cov_matrix[3][3])+ 0.5*cov_matrix[2][3])
    resAndErrs["hgcal"]=[res,resErr]
    
    return_dict[key]=resAndErrs
    
    #return (resAndErrs)

#plot_ratio_single(numerator, denominator, 10, [1,200], label1="TICLv5", xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_v5.png")


def PlotRes(dict_resolutions,tag="test",eta=1.9):

    g_dict={"avg":ROOT.TGraphErrors(),
            "hgcal":ROOT.TGraphErrors(),
            "mtd":ROOT.TGraphErrors()}

    #print(dict_resolutions)
    #print(dict_resolutions[2])
    counter=0
    pts={'x':[],"avg":[],"mtd":[],"hgcal":[],"avgErr":[],"mtdErr":[],"hgcalErr":[]}
    for enkey in dict_resolutions.keys():
        #pts["x"].append(enkey*math.cosh(eta))
        pts["x"].append(enkey)
        for typekey in dict_resolutions[enkey]:
            pts[typekey].append(dict_resolutions[enkey][typekey][0])
            pts[typekey+"Err"].append(dict_resolutions[enkey][typekey][1])
            
            g_dict[typekey].AddPoint(enkey,dict_resolutions[enkey][typekey][0])#,0.,dict_resolutions[enkey][1][typekey])
            g_dict[typekey].SetPointError(counter,0,dict_resolutions[enkey][typekey][1])
        counter=counter+1
    cmg=CMS.cmsCanvas('', 0., 450., 0., 0.04, 'Resolution [ns]', 'p_T [GeV]', square = CMS.kSquare, extraSpace=0.05, iPos=0)
    cmg.cd()
    mg=ROOT.TMultiGraph()


    g_dict["avg"].SetLineColor(ROOT.kMagenta+2)
    g_dict["avg"].SetMarkerColor(ROOT.kMagenta+2)
    g_dict["avg"].SetMarkerStyle(22)
    g_dict["avg"].SetTitle("Combination")

    g_dict["hgcal"].SetLineColor(ROOT.kCyan+2)
    g_dict["hgcal"].SetMarkerColor(ROOT.kCyan+2)
    g_dict["hgcal"].SetMarkerStyle(20)
    g_dict["hgcal"].SetTitle("HGCal time")


    g_dict["mtd"].SetLineColor(ROOT.kSpring+4)
    g_dict["mtd"].SetMarkerColor(ROOT.kSpring+4)
    g_dict["mtd"].SetMarkerStyle(34)
    g_dict["mtd"].SetTitle("MTD time")

    [mg.Add(g_dict[x]) for x in g_dict.keys()]

    mg.SetTitle("Time resolution;E [GeV]; Resolution [ns]")
    mg.Draw("AP")

    legend=ROOT.TLegend(0.2,0.8,0.4,0.7)

    legend.AddEntry(g_dict["avg"])
    legend.AddEntry(g_dict["mtd"])
    legend.AddEntry(g_dict["hgcal"])
    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC(1)  # Imposta le coordinate in termini di frazione del canvas (0-1)
    text.SetTextSize(0.04)  # Imposta la dimensione del testo
    text.SetTextAlign(31)  # Allinea il testo a destra rispetto alla posizione specificata
    text.DrawLatex(0.9, 0.92, "#eta=1.9")  # (x, y, testo) posizione nell'angolo in alto a destra


    mg.SetMinimum(0.)
    mg.SetMaximum(0.040)

    cmg.Draw()
    #CMS.cmsDraw(mg,"AP")
    #cmg.SaveAs(OutputDir + "provares.png")
    #plt.savefig(OutputDir + tag+".png")
    plt.clf()
    plt.errorbar(pts["x"],pts["avg"],pts["avgErr"],label="Combination",color="black",marker="^",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.errorbar(pts["x"],pts["mtd"],pts["mtdErr"],label="ETL",color="red",marker="P",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.errorbar(pts["x"],pts["hgcal"],pts["hgcalErr"],label="HGCAL",color="blue",marker="o",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    hep.cms.lumitext("Phase-II")

    # questa riga solo se plotti con plt e non con ax
    ax = plt.gca()

    for txt in ax.texts:
        if "Phase-II" in txt.get_text():
            txt.set_fontstyle('italic')
        
    plt.text(60, 0.002, "$\pi^\pm$ PU = 0\n$|\eta|="+str(eta)+"$")
    plt.legend()

    plt.xlabel("Simulated $p_T$ [GeV]")
    plt.ylabel("Resolution [ns]")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"mpl.png")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"mpl.pdf")






#print(ReadFileAndFit(fileV5))


#folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV"]


if __name__ == "__main__":


    folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV",
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
        file_name="/eos/user/t/tipaulet/Local_Energy_Samples/"+folders_1p9[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_1p9[i])

        p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_1p9[i],eta,maxfiles))
        jobs.append(p)
        p.start()

        
    for proc in jobs:
        proc.join()
    
    print(return_dict)

    dict_resolutions=return_dict

    PlotRes(dict_resolutions,tag="pT",eta=eta)
    


    dict_resolutions={}

    
    folders_2p2 = ["SinglePionTiming_2p2_100GeV", "SinglePionTiming_2p2_10GeV",
     "SinglePionTiming_2p2_15GeV", "SinglePionTiming_2p2_2GeV",
     "SinglePionTiming_2p2_30GeV", "SinglePionTiming_2p2_4GeV",
     "SinglePionTiming_2p2_50GeV", "SinglePionTiming_2p2_6GeV",
     "SinglePionTiming_2p2_8GeV"]
     



    eta=2.2
    maxfiles=50
    for i in tqdm(prange(len(folders_2p2))):
        file_name="/eos/user/t/tipaulet/Local_Energy_Samples/"+folders_2p2[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_2p2[i])
        p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    print(return_dict)

    dict_resolutions=return_dict


    PlotRes(dict_resolutions,tag="pT",eta=eta)
    

