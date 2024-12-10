""" Tools to read output from SuperclusteringSampleDumper (the CMSSW dumper to produce samples for DNN training) """
from enum import Enum
import math
from typing import Union
import awkward as ak
import numpy as np
import pandas as pd
import uproot
import hist


FeatureType = Enum('FeatureType', ['Seed', 'Candidate', 'Pair'])

feature_nBins = 100
maxEta = 3.2
maxEnergy = 1000.
maxPt = 200.
feature_axises_list = [
    (hist.axis.Regular(feature_nBins, -0.1, 0.1, name="DeltaEtaBaryc", label="Delta eta (seed-candidate) of barycenters"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, -0.5, 0.5, name="DeltaPhiBaryc", label="Delta phi (seed-candidate) of barycenters"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., maxEnergy, name="multi_en", label="Candidate trackster energy (GeV)"), FeatureType.Seed),
    (hist.axis.Regular(feature_nBins, -maxEta, maxEta, name="multi_eta", label="Candidate trackster eta"), FeatureType.Candidate),
    (hist.axis.Regular(feature_nBins, 0., maxPt, name="multi_pt", label="Candidate trackster pT (GeV)"), FeatureType.Candidate),
    (hist.axis.Regular(feature_nBins, -maxEta, maxEta, name="seedEta", label="Seed trackster eta"), FeatureType.Seed),
    (hist.axis.Regular(feature_nBins, -math.pi, math.pi, name="seedPhi", label="Seed trackster phi"), FeatureType.Seed),
    (hist.axis.Regular(feature_nBins, 0., maxEnergy, name="seedEn", label="Seed trackster energy (GeV)"), FeatureType.Seed),
    (hist.axis.Regular(feature_nBins, 0., maxPt, name="seedPt", label="Seed trackster pT (GeV)"), FeatureType.Seed),
    (hist.axis.Regular(feature_nBins, 0., math.pi, name="theta", label="Angle between seed and candidate (theta, °)"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., math.pi, name="theta_xz_seedFrame", label="theta_xz_seedFrame"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., math.pi, name="theta_yz_seedFrame", label="theta_yz_seedFrame"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., math.pi, name="theta_xy_cmsFrame", label="theta_xy_cmsFrame"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., math.pi, name="theta_yz_cmsFrame", label="theta_yz_cmsFrame"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., math.pi, name="theta_xz_cmsFrame", label="theta_xz_cmsFrame"), FeatureType.Pair),
    (hist.axis.Regular(feature_nBins, 0., 1000., name="explVar", label="Explained variance of candidate"), FeatureType.Candidate),
    (hist.axis.Regular(feature_nBins, 0., 1., name="explVarRatio", label="Explained variance ratio of candidate"), FeatureType.Candidate),
]

feature_axises = {axis[0].name : axis[0] for axis in feature_axises_list}
feature_types = {axis[0].name : axis[1] for axis in feature_axises_list}

featureNamesByType_withoutPrefix = {featureType_wanted : [name for name, featureType in feature_types.items() if featureType == featureType_wanted] for featureType_wanted in FeatureType}
featureNamesByType_withPrefix = {featureType_wanted : ["feature_"+name for name, featureType in feature_types.items() if featureType == featureType_wanted] for featureType_wanted in FeatureType}

def zipDNNSample(ar:ak.Array):
    return ak.zip({key : ar[key] for key in ar.fields
                        if key not in ["Event"]},
        depth_limit=2, with_name="SeedCandidatePair")


def seedToDataframe(ar:ak.Array) -> pd.DataFrame:
    return ak.to_dataframe(ar, levelname=lambda x : {0:"eventInternal", 1:"endcapInternal"}[x])

def bestSeedAssocScoreProperties(ar:ak.Array):
    """ Gives seed properties for the seed in each endcap that has the best seedTracksterBestAssociationScore (may not be the highest pT seed) """
    sorted_idx = ak.argsort(ar.seedTracksterBestAssociationScore, ascending=True)
    sorted_ar = ar[sorted_idx]
    return ak.concatenate([ak.singletons(ak.firsts(sorted_ar[sorted_ar.feature_seedEta <0])), ak.singletons(ak.firsts(sorted_ar[sorted_ar.feature_seedEta>0]))],
        axis=1)

def highestPtSeedProperties(ar:ak.Array):
    """ Gives seed properties for the seed in each endcap that has the highest pT """
    sorted_idx = ak.argsort(ar.feature_seedPt, ascending=False)
    sorted_ar = ar[sorted_idx]
    neg_endcap = ak.singletons(ak.firsts(sorted_ar[sorted_ar.feature_seedEta <0]))
    neg_endcap["endcap"] = ak.broadcast_arrays(neg_endcap.seedTracksterIdx, np.array([-1], dtype="int32"),  right_broadcast=False)[1]
    pos_endcap = ak.singletons(ak.firsts(sorted_ar[sorted_ar.feature_seedEta>0]))
    pos_endcap["endcap"] = ak.broadcast_arrays(pos_endcap.seedTracksterIdx, np.array([1], dtype="int32"),  right_broadcast=False)[1]

    return ak.concatenate([neg_endcap, pos_endcap],
        axis=1)

class DNNSampleReader:
    def __init__(self, file: Union[str, uproot.ReadOnlyDirectory], entry_stop=None, pathInsideFileToTree: str = "superclusteringSampleDumper/superclusteringTraining") -> None:
        try:
            self.tree = file[pathInsideFileToTree]
        except TypeError:
            self.tree = uproot.open(file + ":" + pathInsideFileToTree)

        self.ar = self.tree.arrays(entry_stop=entry_stop)
        self.ar_zipped = zipDNNSample(self.ar)
    
    @property
    def nEvents(self):
        return self.tree.num_entries
    
    @property
    def bestSeedProperties(self):
        return highestPtSeedProperties(self.ar_zipped[["seedTracksterIdx", "seedTracksterBestAssociationScore"]+featureNamesByType_withPrefix[FeatureType.Seed]]) 

    @property
    def pairsWithHighestPtSeed(self):
        # seed_idx is nEvts * 2 * unit32 : seedTracksterIdx with highest pt in each endcap
        seed_idx = highestPtSeedProperties(self.ar_zipped[["seedTracksterIdx", "feature_seedPt", "feature_seedEta"]]).seedTracksterIdx

        # get all pairs matching the seed in each endcap
        neg_endcap = self.ar_zipped[seed_idx[:, 0] == self.ar_zipped.seedTracksterIdx]
        neg_endcap["endcap"] = ak.broadcast_arrays(neg_endcap.seedTracksterIdx, np.array([-1], dtype="int32"),  right_broadcast=False)[1]
        pos_endcap = self.ar_zipped[seed_idx[:, 1] == self.ar_zipped.seedTracksterIdx]
        pos_endcap["endcap"] = ak.broadcast_arrays(pos_endcap.seedTracksterIdx, np.array([1], dtype="int32"),  right_broadcast=False)[1]
        return ak.concatenate([neg_endcap, pos_endcap], axis=1)
