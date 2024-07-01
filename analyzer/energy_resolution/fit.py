""" Cruijff function fits for energy resolution """
from functools import partial
import dataclasses
from dataclasses import dataclass

from scipy.optimize import curve_fit
import numpy as np
import hist
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

@dataclass
class CruijffParam:
    A:float
    """ Amplitude"""
    m:float
    """ Central value """
    sigmaL:float
    """ Left tail sigma """
    sigmaR:float
    """ Right tail sigma """
    alphaL:float
    """ Left tail alpha """
    alphaR:float
#     sigmaEff:float


    @property
    def sigmaAverage(self) -> float:
        return (self.sigmaL + self.sigmaR) / 2

    def makeTuple(self) -> tuple[float]:
        return dataclasses.astuple(self)

@dataclass
class CruijffFitResult:
    params:CruijffParam
    covMatrix:np.ndarray

def cruijff(x, A, m, sigmaL,sigmaR, alphaL, alphaR):
    dx = (x-m)
    SL = np.full(x.shape, sigmaL)
    SR = np.full(x.shape, sigmaR)
    AL = np.full(x.shape, alphaL)
    AR = np.full(x.shape, alphaR)
    sigma = np.where(dx<0, SL,SR)
    alpha = np.where(dx<0, AL,AR)
    if(sigmaL == sigmaR):
        sigma = sigmaL
    f = 2*sigma*sigma + alpha*dx*dx
    return A* np.exp(-dx*dx/f)

def histogram_quantiles(h:hist.Hist, quantiles):
    """ Compute quantiles from histogram. Quantiles should be a float (or array of floats) representing the requested quantiles (in [0, 1])
    Returns array of quantile values
    """
    # adapated from https://stackoverflow.com/a/61343915
    assert len(h.axes) == 1 and h.storage_type == hist.storage.Double, "Histogram quantiles needs a 1D double (non-weighted) histogram"
    cdf = (np.cumsum(h.values()) - 0.5 * h.values()) / np.sum(h.values())
    return np.interp(quantiles, cdf, h.axes[0].centers)

def fitCruijff(h_forFit:hist.Hist) -> CruijffFitResult:
    mean = np.average(h_forFit.axes[0].centers, weights=h_forFit.values())
    q_min2, q_min1, median, q_plus1, q_plus2 = histogram_quantiles(h_forFit, [0.5-0.95/2, 0.5-0.68/2, 0.5, 0.5+0.68/2, 0.5+0.95/2])

    # Approximate sigmaL and sigmaR using quantiles. Using quantiles that are equivalent to 1 sigma left and right if the distribution is Gaussian
    # Compared to using standard deviation, it is asymmetric and less sensitive to tails

    p0 = [
        np.max(h_forFit)*0.8, # normalization. The 0.8 is because it seems that the max value is usually a bit higher
        mean, # central value
        median-q_min1, #sigmaL : this quantile difference is 1sigma for a Gaussian
        q_plus1-median, # sigmaR
        (q_min1-q_min2) / (median-q_min1)/3.81 * 0.28067382, #alphaL : in the ratio, numerator and denominator should be sigma for a gaussian. Otherwise, the heavier the tails, the higher from one. Then some norm coefficient (could be improved)
        (q_plus2-q_plus1) / (q_plus1-median)/3.81 * 0.28067382 #alphaR
    ]
    try:
        param_optimised,param_covariance_matrix = curve_fit(cruijff, h_forFit.axes[0].centers, h_forFit.values(),
            p0=p0, sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8), absolute_sigma=True, maxfev=500000,
            bounds=np.transpose([(0., np.inf), (-np.inf, np.inf), (0., np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
            )
    except ValueError: # sometimes it fails with ValueError: array must not contain infs or NaNs and removing bounds helps
        param_optimised,param_covariance_matrix = curve_fit(cruijff, h_forFit.axes[0].centers, h_forFit.values(),
            p0=p0, sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8), absolute_sigma=True, maxfev=500000,
            #bounds=np.transpose([(0., np.inf), (-np.inf, np.inf), (0., np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
            )
    return CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix)


eratio_axis = partial(hist.axis.Regular, 500, 0, 2, name="e_ratio")
eta_axis = hist.axis.Variable([1.65, 2.15, 2.75], name="absSeedEta", label="|eta|seed")
seedPt_axis = hist.axis.Variable([ 0.44310403, 11.58994007, 23.00519753, 34.58568954, 46.85866928,
       58.3225441 , 68.96975708, 80.80027771, 97.74741364], name="seedPt", label="Seed Et (GeV)") # edges are computed so that there are the same number of events in each bin
def make_scOrTsOverCP_energy_histogram(name, label=None):
    h = hist.Hist(eratio_axis(label=label),
                  eta_axis, seedPt_axis, name=name, label=label)
    return h

def fill_scOverCP_energy_histogram(h:hist.Hist, df:pd.DataFrame, minEn:float, maxEn:float):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
    dfN = df.where((df.regressed_energy_CP >= minEn) & (df.regressed_energy_CP <= maxEn)).dropna()
    h.fill(e_ratio=dfN.raw_energy/dfN.regressed_energy_CP)

def fill_seedTsOverCP_energy_histogram(h:hist.Hist, df:pd.DataFrame):
    """ df should be CPtoTs_df ie CaloParticle to seed trackster (highest pt trackster for each endcap) """
    h.fill(e_ratio=df.raw_energy/df.regressed_energy_CP,
        seedPt=df.regressed_energy_CP)

def make_num_eff_histo(name, bins, minVar, maxVar, label=None):
    eratio_eff_axis = partial(hist.axis.Regular, bins, minVar, maxVar, name="e_ratio")
    h = hist.Hist(eratio_eff_axis(label=label), name=name, label=label)
    return h

def fill_num_eff_histo(h:hist.Hist, df:pd.DataFrame, minShared:float, var:str, corrected:bool):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
    if(corrected == False):
        dfN = df.where((df.sharedE / df.raw_energy_CP >= minShared))
    else:
        dfN = df.where((df.sharedE / df.tot_sharedE >= minShared))
    h.fill(getattr(dfN, var))

def fill_den_eff_histo(h:hist.Hist, df:pd.DataFrame, minShared:float, var:str):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
#     print(getattr(df,var))
    h.fill(getattr(df, var))


def fitMultiHistogram(h:list[hist.Hist], sigmaEff:bool=False) -> list[list[CruijffFitResult]]:
    """ Cruijff fit of multi-dimensional histogram of Supercluster/CaloParticle energy """
    res = []
    sigmasEff = []
#     for eta_bin in range(len(h[0].axes["absSeedEta"])):

    for i in range(len(h)):
        h_1d = h[i]
        res.append([])
        sigmasEff.append([])
#         for seedPt_bin in range(len(h)):
#             print(eta_bin, seedPt_bin)
#             h_1d = h[seedPt_bin][{"absSeedEta":eta_bin, "seedPt":0}]
        fitR, sEff = fitCruijff(h_1d, sigmaEff)
        res[-1] = fitR
        sigmasEff[-1] = sEff
    return (res, sigmasEff)

def plotSingleHistWithFit(h_1d:hist.Hist, fitRes:CruijffFitResult, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    hep.histplot([h_1d], label=["Best associated trackster"], ax=ax, yerr=False, flow="none")
    x_plotFct = np.linspace(h_1d.axes[0].centers[0], h_1d.axes[0].centers[-1], 500)
    # print(fitRes)
#     sigma = fitRes.params.sigmaEff if fitRes.params.sigmaEff != 0 else fitRes.params.sigmaAverage
    # ax.plot(x_plotFct, cruijff(x_plotFct,*fitRes.params.makeTuple()), label=f"Cruijff fit\n$\sigma={fitRes.params.sigmaAverage:.3f}$")
    # ax.set_xlim(0.1, 3.5)
    # ax.set_ylabel("Events")
    # ax.legend()
    hep.cms.text("Preliminary", exp="TICLv5", ax=ax)
    hep.cms.lumitext("PU=0", ax=ax)

bin_edgesEnergy = [(9.5, 10.5), (19.5, 20.5), (29.5, 30.5), (49.5, 50.5), (69.5, 70.5), (99.5, 100.5), (199.5, 200.5), (399.5, 400.5), (599.5, 600.5)]

def ptBinToText(ptBin:int) -> str:
    low, high = bin_edgesEnergy[ptBin]
    return r"$E_{\text{gen}} \in \left[" + f"{low:.3g}; {high:.3g}" + r"\right]$"

def etaBinToTextEqual(etaBin:float) -> str:
    return r"$|\eta_{\text{gen}}| =" + f"{etaBin}$"
def plotAllFits(h:list[hist.Hist], fitResults:list[CruijffFitResult], etaFloat:float, bin_edgesEnergy, outputDir):

    for i in range(len(h)):
        seedPt_binT = bin_edgesEnergy[i]
        h_1d = h[i]
        eta_bins =  (1.89,1.91)
        eta_bin = 0
        plotSingleHistWithFit(h_1d, fitResults[0][i])
        plt.text(0.05, 0.95, etaBinToTextEqual(etaFloat)+"\n"+ptBinToText(i), va="top", transform=plt.gca().transAxes, fontsize=20)
        plt.savefig(f"{outputDir}/{seedPt_binT[0]}_{seedPt_binT[1]}_{eta_bin}.png")
