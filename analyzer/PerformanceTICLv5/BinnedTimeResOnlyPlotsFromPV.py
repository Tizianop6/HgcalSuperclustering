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


#fileV5 = "/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PUOnlyPV/"
# fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/TimeResolution/SinglePionTiming_2p2_100GeV/histo/"

OutputDir = "/eos/user/t/tipaulet/www/BinnedTimeResPUOnlyPV/"

create_directory(OutputDir)

#@njit(parallel=True)

#print(ReadFileAndFit(fileV5))

#folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV"]

def CreatePlots(dict_residuals,eta):
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

    rng_histos=0.2
    histedges_np = np.array(histedges)
    for key in histdict:
        for i in range(len(histedges)-1):
            center = (histedges[i]+histedges[i+1])/2
            rng = (histedges[i+1]-histedges[i])/2
            histdict[key].append(hist.Hist(hist.axis.Regular(100,-rng_histos,rng_histos,name="Residual [ns]"), name=str(i)+str(key), label=str(i)+str(key)))
        #histdict[key].append(hist.Hist(hist.axis.Regular(100,-0.1,0.1,name="tres"), name=str(i), label=str(i)+"overflow"))


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
            if keyDet=="avg":
                histdict[keyDet][i].plot(label="Combination",histtype='step', color="black")
            if keyDet=="hgcal":
                histdict[keyDet][i].plot(label="HGCAL",histtype='step',color="blue")
            if keyDet=="mtd":
                histdict[keyDet][i].plot(label="ETL",histtype='step',color="red")
        plt.text(0.01, 3, "["+"{:.2f}".format(histedges[i])+","+"{:.2f}".format(histedges[i+1])+"] GeV")
        plt.legend()
        plt.savefig(OutputDir+"Histo"+"{:02d}".format(i)+"eta"+str(eta)+".png")

    fitres={}
    for keyDet in histdict:
        fitres[keyDet]=fitMultiHistogram(histdict[keyDet])

    for i in range(len(histedges)-1):
        plt.clf()
        xlist=np.linspace(-rng_histos,rng_histos,1000)
        for keyDet in histdict:
            if keyDet=="avg":
                histdict[keyDet][i].plot(label="Combination",histtype='step', color="black")
            if keyDet=="hgcal":
                histdict[keyDet][i].plot(label="HGCAL",histtype='step',color="blue")
            if keyDet=="mtd":
                histdict[keyDet][i].plot(label="ETL",histtype='step',color="red")
            #histdict[keyDet][i].plot()
            try:
                cruj=partial(cruijff, A=fitres[keyDet][0][i].params.A, m=fitres[keyDet][0][i].params.m, sigmaL=fitres[keyDet][0][i].params.sigmaL,
                sigmaR=fitres[keyDet][0][i].params.sigmaR, alphaL=fitres[keyDet][0][i].params.alphaL, alphaR=fitres[keyDet][0][i].params.alphaR)
                if keyDet=='avg':
                    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color="black")#,color = "xkcd:magenta")
                if keyDet=='hgcal':
                    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color="blue")#,color = "xkcd:magenta")
                if keyDet=='mtd':
                    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color="red")#,color = "xkcd:magenta")
                
                plt.text(0.01, 3, "["+"{:.2f}".format(histedges[i])+","+"{:.2f}".format(histedges[i+1])+"] GeV")
            except AttributeError:
                plt.text(0.01, 0.01, "["+"{:.2f}".format(histedges[i])+","+"{:.2f}".format(histedges[i+1])+"] GeV")
        plt.legend()
        plt.savefig(OutputDir+"Fit_"+"{:02d}".format(i)+"eta"+str(eta)+".png")
    """
    plt.clf()
    hep.hist2dplot(hist2d)

    plt.show()
    #plt.pcolormesh(hist2d.axes.edges.T, hist2d.values().T)

    plt.savefig(OutputDir+"Hist2d.png")
    """



    dict_pts={'x':[],'y':[],'xErr':[], 'mtd':[],'mtdErr':[],'hgcal':[],'hgcalErr':[],'avg':[],'avgErr':[]}

    for i in range(len(histedges)-1):

        center = (histedges[i]+histedges[i+1])/2
        rng = (histedges[i+1]-histedges[i])/2

        dict_pts["x"].append(center)
        dict_pts["xErr"].append(rng)
        badfit=False
        for keyDet in histdict:
            try:
                cov_matrix=fitres[keyDet][0][i].covMatrix
                resErr=math.sqrt(0.25*(cov_matrix[2][2]+cov_matrix[3][3])+ 0.5*cov_matrix[2][3])
                if resErr>0.01:
                    badfit=True
                dict_pts[keyDet].append((fitres[keyDet][0][i].params.sigmaL+fitres[keyDet][0][i].params.sigmaR)/(2))
                dict_pts[keyDet+"Err"].append(resErr)
            except AttributeError:
                dict_pts[keyDet].append(-1.)
                dict_pts[keyDet+"Err"].append(0)
        if badfit==True:
            for keyDet in histdict:
                dict_pts[keyDet][-1]=-1
                dict_pts[keyDet+"Err"][-1]=0



    


    plt.clf()
    plt.errorbar(dict_pts["x"],dict_pts["avg"],dict_pts["avgErr"],dict_pts["xErr"],label="Combination",color="black",marker="^",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.errorbar(dict_pts["x"],dict_pts["mtd"],dict_pts["mtdErr"],dict_pts["xErr"],label="ETL",color="red",marker="P",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.errorbar(dict_pts["x"],dict_pts["hgcal"],dict_pts["hgcalErr"],dict_pts["xErr"],label="HGCAL",color="blue",marker="o",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.ylim(0, 0.045)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 200PU\n$|\eta|="+str(eta)+"$")
    plt.legend()

    plt.xlabel("Simulated Energy [GeV]")
    plt.ylabel("Resolution [ns]")
    plt.savefig(OutputDir +"eta"+str(eta)+"mpl.png")

    #Same graph without x-axis error bars
    plt.clf()
    plt.errorbar(dict_pts["x"],dict_pts["avg"],dict_pts["avgErr"],label="Combination",color="black",marker="^",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.errorbar(dict_pts["x"],dict_pts["mtd"],dict_pts["mtdErr"],label="ETL",color="red",marker="P",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.errorbar(dict_pts["x"],dict_pts["hgcal"],dict_pts["hgcalErr"],label="HGCAL",color="blue",marker="o",linestyle='',capsize=3,capthick=2,markersize=6*1.5)
    plt.ylim(0, 0.045)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 200PU\n$|\eta|="+str(eta)+"$")
    plt.legend()

    plt.xlabel("Simulated Energy [GeV]")
    plt.ylabel("Resolution [ns]")
    plt.savefig(OutputDir +"eta"+str(eta)+"mpl_woerr.png")



if __name__ == "__main__":

    
    folders_2p2 = ["SinglePionTimingPU_2p2_100GeV", "SinglePionTimingPU_2p2_10GeV",
     "SinglePionTimingPU_2p2_15GeV", "SinglePionTimingPU_2p2_2GeV",
     "SinglePionTimingPU_2p2_30GeV", "SinglePionTimingPU_2p2_4GeV",
     "SinglePionTimingPU_2p2_50GeV", "SinglePionTimingPU_2p2_6GeV",
     "SinglePionTimingPU_2p2_8GeV"]

    #manager = multiprocessing.Manager()
    #return_dict = manager.dict()
    #jobs = [] 
    return_dict={}

    eta=2.2
    maxfiles=1
    dict_residuals={}
    for i in tqdm(range(len(folders_2p2))):
        file_name="/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PU/"+folders_2p2[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_2p2[i])
        #dict_residuals.update(ReadFileAndFit(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        #print(dict_residuals)
        with open('files_onlyPV/residuals_'+match.group(1)+'_'+folders_2p2[i]+'.pkl', 'rb') as file:
            data=pickle.load(file)
        #print(data)
        dict_residuals[folders_2p2[i]]=data

        #p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        #jobs.append(p)
        #p.start()
        #p.join()
    CreatePlots(dict_residuals,eta=2.2)
    #for proc in jobs:
    #    proc.join()
    

    folders_1p9 = ["SinglePionTimingPU_1p9_100GeV", "SinglePionTimingPU_1p9_10GeV",
     "SinglePionTimingPU_1p9_15GeV", "SinglePionTimingPU_1p9_2GeV",
     "SinglePionTimingPU_1p9_30GeV", "SinglePionTimingPU_1p9_4GeV",
     "SinglePionTimingPU_1p9_50GeV", "SinglePionTimingPU_1p9_6GeV",
     "SinglePionTimingPU_1p9_8GeV"]

    
    eta=1.9
    maxfiles=1
    dict_residuals={}
    for i in tqdm(range(len(folders_1p9))):
        file_name="/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PU/"+folders_1p9[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_1p9[i])
        #dict_residuals.update(ReadFileAndFit(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        #print(dict_residuals)
        with open('files_onlyPV/residuals_'+match.group(1)+'_'+folders_1p9[i]+'.pkl', 'rb') as file:
            data=pickle.load(file)
        #print(data)
        dict_residuals[folders_1p9[i]]=data

        #p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        #jobs.append(p)
        #p.start()
        #p.join()
    CreatePlots(dict_residuals,eta=1.9)


    #dict_residuals=return_dict
    
    #with open('residuals.pkl', 'wb') as handle:
    #    pickle.dump(dict_residuals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """

    #manager = multiprocessing.Manager()
    #return_dict = manager.dict()
    #jobs = [] 
    return_dict={}

    eta=1.9
    dict_residuals={}
    for i in tqdm(range(len(folders_1p9))):
        file_name="/afs/cern.ch/user/t/tipaulet/test/TimeResolutionFinal200PU/"+folders_1p9[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_1p9[i])
        #dict_residuals.update(ReadFileAndFit(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        #print(dict_residuals)
        with open('files_onlyPV/residuals_'+match.group(1)+'_'+folders_1p9[i]+'.pkl', 'rb') as file:
            data=pickle.load(file)
        #print(data)
        dict_residuals[folders_1p9[i]]=data

        #p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_2p2[i],eta,maxfiles))
        #jobs.append(p)
        #p.start()
        #p.join()

    CreatePlots(dict_residuals,eta=eta)
    """

    





