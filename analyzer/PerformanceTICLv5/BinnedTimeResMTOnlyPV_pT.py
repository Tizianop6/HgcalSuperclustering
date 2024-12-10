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
import pickle
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


fileV5 = "/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PUOnlyPV/"

OutputDir = "/eos/user/t/tipaulet/www/BinnedTimeResPUOnlyPV/"

create_directory(OutputDir)


@njit
def FindCandidate(tracksters_in_cand,track_in_cand,tid):
    cand_idx = -1
    for i, k in enumerate(track_in_cand):
        if k == -1 : continue
        tracksters=tracksters_in_cand[i]
        if not len(tracksters): continue
        if tracksters[0] == tid:
            cand_idx = i
            break

    return cand_idx
            

def AnalyzeFile(dumper, cands, simCands, ass,dict_singlefiles,counter,eta):

    dict_temp={"avg":[],
                    "hgcal":[],
                    "mtd":[],
                    "cp_energy":[]}

    print("file ", counter)
    for ev in range(len(cands)):
        #tmEv = tms[ev]
        candEv = cands[ev]
        simCandEv = simCands[ev]
        assEv = ass[ev]
        #simEv = sims[ev]
        #tracksEv = tracks[ev]
        for simCand_idx in range(2):

            simTrack = simCandEv.simTICLCandidate_track_in_candidate[simCand_idx]
            if simTrack == -1: 
                continue
            simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
            simRegrEnergy = simCandEv.simTICLCandidate_regressed_energy[simCand_idx]
            #simRegrPt = simCandEv.simTICLCandidate_regressed_pt[simCand_idx]
            simTime = simCandEv.simTICLCandidate_time[simCand_idx]
            
            '''
            tk_idx_in_coll = -1
            try:
                tk_idx_in_coll = np.where(tracksEv.track_id == simTrack)[0][0] 
            except:
                continue
            
            if tk_idx_in_coll == -1 or tracksEv.track_pt[tk_idx_in_coll] < 1 or tracksEv.track_missing_outer_hits[tk_idx_in_coll] > 5 or not tracksEv.track_quality[tk_idx_in_coll]: 
                continue
            ''' 
            start = time.process_time()
            
            sharedE = assEv.ticlCandidate_simToReco_CP_sharedE[simCand_idx]
            score = assEv.ticlCandidate_simToReco_CP_score[simCand_idx]
            if not len(sharedE): continue
            idx_argmin=ak.argmin(score)
            if sharedE[idx_argmin]/simRegrEnergy < 0.3: continue
            simToReco = assEv.ticlCandidate_simToReco_CP[simCand_idx]

            # This is the trackster ID
            tid = simToReco[idx_argmin]

            # Information about tracks in recoTICLCandidate in order to select only charged ones
            track_in_cand=candEv.track_in_candidate
            
            # Find charged recoTICLCandidate that contains the desired trackster
            cand_idx =FindCandidate(candEv.tracksters_in_candidate,candEv.track_in_candidate,tid)
            
            if cand_idx == -1: continue
            

            # HGCAL time & error
            candidate_time = candEv.candidate_time[cand_idx]
            candidate_timeErr= candEv.candidate_timeErr[cand_idx]
            track_in_cand=candEv.track_in_candidate[cand_idx]
            
            #MTD time & error
            candidate_time_MTD = candEv.candidate_time_MTD[cand_idx]
            candidate_time_MTDErr= candEv.candidate_time_MTD_err[cand_idx]
    

            
            t_HGCal = False
            t_MTD = False

            if (candidate_timeErr>0):
                t_HGCal=True 
                dict_temp["hgcal"].append(simTime-candidate_time)
                time_avg=candidate_time
                time_avgErr=candidate_timeErr

            if(candidate_time_MTDErr>0):
                t_MTD=True
                MTD_time=candidate_time_MTD
                dict_temp["mtd"].append(simTime-MTD_time)
                if(t_HGCal):
                    inv_hgcal_err=1./(time_avgErr**2)
                    inv_mtd_err=1./(candidate_time_MTDErr**2)
                    time_avg=(candidate_time*inv_hgcal_err+MTD_time*inv_mtd_err)/(inv_mtd_err+inv_hgcal_err)
                else:
                    time_avg=MTD_time

            if(t_MTD or t_HGCal):
                dict_temp["avg"].append(simTime-time_avg)
                dict_temp["cp_pt"].append(simRegrEnergy/math.cosh(eta))
                dict_temp["cp_energy"].append(simRegrPt)
                if (t_HGCal==False):
                    dict_temp["hgcal"].append(-100.)
                if (t_MTD==False):
                    dict_temp["mtd"].append(-100.)
                    

    dict_singlefiles[counter] = dict_temp
    







def ReadFileAndFit(filename,key,tag="prova",eta=2.2, maxfiles=1):

    # Read input folder
    dumperInputV5 = DumperInputManager([
        filename
    ], limitFileCount=maxfiles)


    # Dictionary with the residuals and the CP simEnergy
    ticl_residuals={"avg":[],
                    "hgcal":[],
                    "mtd":[],
                    "cp_energy":[]}


    # Multithreading
    manager1 = multiprocessing.Manager()
    dict_singlefiles = manager1.dict()
    jobs1 = [] 

    
    for i in range(len(dumperInputV5.inputReaders)):
        dumper = dumperInputV5.inputReaders[i].ticlDumperReader
        cands = dumper.candidates
        simCands = dumper.simCandidates
        ass = dumper.associations
        p1 = multiprocessing.Process(target=AnalyzeFile, args=(dumper, cands, simCands, ass, dict_singlefiles,i,eta))
        jobs1.append(p1)
        p1.start()
        
        
    for proc1 in jobs1:
        proc1.join()
    
    
    for itkey in ticl_residuals:
        for i in range(len(dumperInputV5.inputReaders)):
            #print("dict_singlefiles[i][key]", dict_singlefiles[i][key])
            ticl_residuals[itkey] = ticl_residuals[itkey] + dict_singlefiles[i][itkey]

    with open('files_onlyPV_Pt/residuals_'+str(key)+'_'+str(tag)+'.pkl', 'wb') as handle:
        pickle.dump(ticl_residuals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print ("ticl_residuals",ticl_residuals)

    return_dict[key]=ticl_residuals
    #print(return_dict)
    return return_dict






#print(ReadFileAndFit(fileV5))


#folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV"]

if __name__ == "__main__":


    folders_2p2 = ["SinglePionTimingPU_2p2_100GeV", "SinglePionTimingPU_2p2_10GeV",
     "SinglePionTimingPU_2p2_15GeV", "SinglePionTimingPU_2p2_2GeV",
     "SinglePionTimingPU_2p2_30GeV", "SinglePionTimingPU_2p2_4GeV",
     "SinglePionTimingPU_2p2_50GeV", "SinglePionTimingPU_2p2_6GeV",
     "SinglePionTimingPU_2p2_8GeV"]

    folders_1p9 = ["SinglePionTimingPU_1p9_100GeV", "SinglePionTimingPU_1p9_10GeV",
     "SinglePionTimingPU_1p9_15GeV", "SinglePionTimingPU_1p9_2GeV",
     "SinglePionTimingPU_1p9_30GeV", "SinglePionTimingPU_1p9_4GeV",
     "SinglePionTimingPU_1p9_50GeV", "SinglePionTimingPU_1p9_6GeV",
     "SinglePionTimingPU_1p9_8GeV"]

    """
    dict_resolutions={}

    return_dict={}

    eta=2.2
    maxfiles=50
    dict_residuals={}
    for i in tqdm(range(len(folders_2p2))):
        file_name="/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PU/"+folders_2p2[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_2p2[i])
        dict_residuals.update(ReadFileAndFit(file_name,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        print(dict_residuals)
    


    """
    
    dict_resolutions={}

    return_dict={}

    eta=1.9
    maxfiles=50
    dict_residuals={}
    for i in tqdm(range(len(folders_1p9))):
        file_name="/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PU/"+folders_1p9[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_1p9[i])
        dict_residuals.update(ReadFileAndFit(file_name,int(match.group(1)),folders_1p9[i],eta,maxfiles))
        print(dict_residuals)


    os.system("python3 BinnedTimeResOnlyPlots.py")




    """
    scale=math.cosh(eta)
    hist2d_dict =  {'avg': hist.Hist(hist.axis.Regular(10, 0*scale, 120*scale,name="reco", label="recoPt simPt Combination[GeV] ")),
                    'mtd': hist.Hist(hist.axis.Regular(10, 0*scale, 120*scale,name="reco", label="recoPt simPt ETL [GeV]")),
                    'hgcal': hist.Hist(hist.axis.Regular(10, 0*scale, 120*scale,name="reco", label="recoPt simPt HGCAL [GeV] "))}
    
    histdict = {'avg':[],
                'mtd':[],
                'hgcal':[]}


    histedges = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,9,11,19,25,40,60,90,110]
    #scaling for pt->E
    for i in range(len(histedges)):
        histedges[i]=histedges[i]*scale
    avg=np.zeros(len(histedges))
    counts=np.zeros(len(histedges))

    histedges_np = np.array(histedges)
    for key in histdict:
        for i in range(len(histedges)-1):
            center = (histedges[i]+histedges[i+1])/2
            rng = (histedges[i+1]-histedges[i])/2
            histdict[key].append(hist.Hist(hist.axis.Regular(100,-rng+center,rng+center,name="tres"), name=str(i)+str(key), label=str(i)+str(key)))
        histdict[key].append(hist.Hist(hist.axis.Regular(100,histedges[-1],rng+histedges[-1],name="tres"), name=str(i), label=str(i)+"overflow"))


    for keyDet in histdict:
        for key in dict_residuals:
            #hist2d_dict[keyDet].fill(dict_residuals[key][keyDet],dict_residuals[key]["cp_energy"])
            reco_np=np.array(dict_residuals[key][keyDet])
            sim_np=np.array(dict_residuals[key]["cp_energy"])
            for i in range(len(histedges)-1):
                mask=(sim_np > float(histedges[i])) & (sim_np < float(histedges[i+1]))
                #avg[i]=avg[i]+np.sum(reco_np[mask])
                #counts[i]=counts[i]+len(reco_np[mask])
                
                histdict[keyDet][i].fill(reco_np[mask])
                #histlist[int((dict_residuals[key][1][i]-sim_min)/sim_binw)].fill(dict_residuals[key][0][i])



    for i in range(len(histedges)-1):
        plt.clf()

        for keyDet in histdict:
            histdict[keyDet][i].plot()
        plt.legend()
        plt.savefig(OutputDir+"OnlyHisto"+str(i)+".png")

    fitres={}
    for keyDet in histdict:
        fitres[keyDet]=fitMultiHistogram(histdict[keyDet])

    for i in range(len(histedges)-1):
        plt.clf()
        xlist=np.linspace(histedges[i],histedges[i+1],1000)
        for keyDet in histdict:
            histdict[keyDet][i].plot()
            cruj=partial(cruijff, A=fitres[keyDet][0][i].params.A, m=fitres[keyDet][0][i].params.m, sigmaL=fitres[keyDet][0][i].params.sigmaL,
            sigmaR=fitres[keyDet][0][i].params.sigmaR, alphaL=fitres[keyDet][0][i].params.alphaL, alphaR=fitres[keyDet][0][i].params.alphaR)
            plt.plot(xlist,cruj(np.array(xlist)),linewidth=4)#,color = "xkcd:magenta")

        plt.legend()
        plt.savefig(OutputDir+str(i)+".png")
    "
    plt.clf()
    hep.hist2dplot(hist2d)

    plt.show()
    #plt.pcolormesh(hist2d.axes.edges.T, hist2d.values().T)

    plt.savefig(OutputDir+"Hist2d.png")
    



    dict_pts={'x':[],'y':[], 'mtd':[],'mtdErr':[],'hgcal':[],'hgcalErr':[],'avg':[],'avgErr':[]}

    for i in range(len(histedges)-1):

        center = (histedges[i]+histedges[i+1])/2
        rng = (histedges[i+1]-histedges[i])/2
        dict_pts["x"].append(center)
        dict_pts["xErr"].append(rng)

        for keyDet in histdict:
            cov_matrix=fitres[keyDet][0][i].covMatrix
            resErr=math.sqrt(0.25*(cov_matrix[2][2]+cov_matrix[3][3])+ 0.5*cov_matrix[2][3])
            dict_pts[keyDet].append((fitres[keyDet][0][i].params.sigmaL+fitres[keyDet][0][i].params.sigmaR)/(2))
            dict_pts[keyDet+"Err"].append(resErr)


    


    plt.clf()
    plt.errorbar(dict_pts["x"],dict_pts["avg"],dict_pts["avgErr"],dict_pts["xErr"],label="Combination",color="xkcd:magenta",marker="^",linestyle='',capsize=3,capthick=2)
    plt.errorbar(dict_pts["x"],dict_pts["mtd"],dict_pts["mtdErr"],dict_pts["xErr"],label="ETL",color="xkcd:azure",marker="P",linestyle='',capsize=3,capthick=2)
    plt.errorbar(dict_pts["x"],dict_pts["hgcal"],dict_pts["hgcalErr"],dict_pts["xErr"],label="HGCAL",color="xkcd:green",marker="o",linestyle='',capsize=3,capthick=2)
    plt.ylim(0, 0.06)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 200PU\n$\eta="+str(eta)+"$")
    plt.legend()

    plt.xlabel("Simulated Energy [GeV]")
    plt.ylabel("Resolution [ns]")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"mpl.png")
    """



