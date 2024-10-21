""" Reader class to read TICLDumper output """
from functools import cached_property
from typing import Literal
from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from .assocs import *
from .tracksters import *
from .recoToSim import *
from .simToReco import *

from typing import List

import awkward as ak
print(ak.__version__)

def superclustersToEnergy(supercls_ts_idxs:ak.Array, tracksters:ak.Array) -> ak.Array:
    """ Computes the total energy of a supercluster from an array of supercluster ids
    Preserves the inner index (usually CP id)
    Parameters :
     - supercls_ts_idxs : type nevts * var * var * uint64
    Returns : type nevts * var * float (energy sum)
    """
    # FIrst flatten the inner dimension (CP id) before taking tracksters
    energies_flat = tracksters.raw_energy[ak.flatten(supercls_ts_idxs, axis=-1)]
    # Rebuild the inner index
    energies = ak.unflatten(energies_flat,  ak.flatten(ak.num(supercls_ts_idxs, axis=-1)), axis=-1)

    return ak.sum(energies, axis=-1)




class DumperReader:
    """ Reads TICLDumper output """
    class MultiFileReader:
        """ From a list of files opened with uproot, find the the first one which has the requested key (for finding the file having the ticlDumper among different files holding the same events) """
        def __init__(self, files:list[uproot.ReadOnlyDirectory]):
            self.files = files

        def __getitem__(self, key):
            excpts = []
            for file in self.files:
                try:
                    return file[key]
                except uproot.KeyInFileError as e:
                    excpts.append(e)
            raise KeyError(*excpts)

    def __init__(self, file: Union[str, uproot.ReadOnlyDirectory, List[uproot.ReadOnlyDirectory]], directoryName: str = "ticlDumper") -> None:
        """ file can be :
         - a str : opens the file and loads ticlDumper
         - an already opened uproot file
         - a list of uproot files, in which case it opens them in order and uses the first one that has ticlDumper in it
        """
        try:
            self.fileDir = file[directoryName]
        except TypeError:
            try:
                self.fileDir = self.MultiFileReader([f[directoryName] for f in file])
            except TypeError:
                self.fileDir = uproot.open(file + ":" + directoryName)

    @property
    def nEvents(self):
        return self.fileDir["trackstersCLUE3DHigh"].num_entries

    @cached_property
    def tracksters(self) -> ak.Array:
        return self.fileDir["trackstersCLUE3DHigh"].arrays()

    @cached_property
    def tracksters_zipped(self) -> ak.Array:
        return ak.zip({"ts_id" : ak.local_index(self.tracksters.raw_energy, axis=1)} |
                      {key : self.tracksters[key] for key in self.tracksters.fields
                        if key not in ["vertices_multiplicity","event", "NClusters", "NTracksters"]},
            depth_limit=2, # don't try to zip vertices
            with_name="tracksters"
        )
    @cached_property
    def trackstersMerged(self) -> ak.Array:
        return self.fileDir["trackstersTiclCandidate"].arrays()

    @cached_property
    def trackstersMergedv4(self) -> ak.Array:
        return self.fileDir["trackstersmerged"].arrays()

    @cached_property
    def candidates(self) -> ak.Array:
        return self.fileDir["candidates"].arrays()

    @cached_property
    def simCandidates(self) -> ak.Array:
        return self.fileDir["simTICLCandidate"].arrays()

    @cached_property
    def tracks(self) -> ak.Array:
        return self.fileDir["tracks"].arrays()

    @cached_property
    def trackstersMerged_zipped(self) -> ak.Array:
        # Create the base dictionary
        base_dict = {"ts_id": ak.local_index(self.trackstersMerged.raw_energy, axis=1)}
        # Update the base dictionary with the other fields
        base_dict.update({key: self.trackstersMerged[key] for key in self.trackstersMerged.fields
                          if key not in ["vertices_multiplicity", "event", "NClusters", "NTracksters"]})
        return ak.zip(
            base_dict,
            depth_limit=2,  # don't try to zip vertices
            with_name="trackstersMerged"
        )



    @cached_property
    def simTrackstersCP(self) -> ak.Array:
        return self.fileDir["simtrackstersCP"].arrays(filter_name=["event_", "raw_energy", "raw_energy_em", "regressed_energy","regressed_pt", "barycenter_*",'trackIdx'])
    @cached_property
    def simTrackstersCP_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.simTrackstersCP, levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x])



    @cached_property
    def simTrackstersSC(self) -> ak.Array:
        return self.fileDir["simtrackstersSC"].arrays(filter_name=["event_", "raw_energy", "raw_energy_em", "regressed_energy","regressed_pt", "barycenter_*",'trackIdx'])
    @cached_property
    def simTrackstersSC_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.simTrackstersSC, levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x])


    @cached_property
    def superclusters(self) -> ak.Array:
        """ Gets the supercluster trackster ids (since ticlv5 this actually includes also tracksters in one-trackster superclusters)
        type: nevts * var (superclsCount) * var (trackstersInSupercls) * uint64 (trackster id)
        """
        return self.fileDir["superclustering/linkedResultTracksters"].array()

    @cached_property
    def superclusters_all(self) -> ak.Array:
        """ Supercluster tracksters ids. Tracksters not in a supercluster are included in a one-trackster supercluster each
        Since ticlv5 this is actually identical to superclusters
        """
        #return self.fileDir["superclustering/superclusteredTrackstersAll"].array()
        return self.fileDir["superclustering/linkedResultTracksters"].array()

    @cached_property
    def superclusteringDnnScore(self) -> ak.Array:
        ar =  self.fileDir["superclustering/superclusteringDNNScore"].array()
        return ak.zip({"ts_seed":ar["superclusteringDNNScore._0"], "ts_cand":ar["superclusteringDNNScore._1"], "dnnScore":ak.values_astype(ar["superclusteringDNNScore._2"], np.single)/65535.}, with_name="dnnInferencePair")

    @cached_property
    def associations(self) -> ak.Array:
        return self.fileDir["associations"].arrays(filter_name=["event_", "tsCLUE3D_*", 'ticlCandidate_*', 'Mergetracksters_*'])

    #@cached_property
    #def TICLCandidates(self) -> ak.Array:
    #    return self.fileDir["candidates"].arrays(filter_name=["event_", "candidate_charge", 'candidate_time', 'candidate_energy', 'candidate_raw_energy', 'tracksters_in_candidate','track_in_candidate'])


    @cached_property
    def supercluster_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.superclusters, anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])

    @cached_property
    def supercluster_all_df(self) -> pd.DataFrame:
        """ Same as superclusters_df but tracksters not in a supercluster are included in a one-trackster supercluster each
        """
        return self.supercluster_df
        # return ak.to_dataframe(self.superclusters_all, anonymous="ts_id",
        #     levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])

    @cached_property
    def supercluster_merged_properties_all(self) -> pd.DataFrame:
        """ Dataframe holding supercluster properties (one row per supercluster)

        Tracksters not in a supercluster are included as one-trackster superclusters
        """
        return (supercluster_joinTracksters(self.supercluster_all_df, self.tracksters_zipped[["raw_energy", "regressed_energy"]])
                .groupby(level=["eventInternal", "supercls_id"])
                .agg(
                    raw_energy_supercls=pd.NamedAgg("raw_energy", "sum"),
                    regressed_energy_supercls=pd.NamedAgg("regressed_energy", "sum"),
        ))

    @cached_property
    def assocs_bestScore_simToReco_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster

        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToReco(self.associations))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )

    def assocs_bestScore_recoToSim_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster

        Parameters :
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )

    @cached_property
    def assocs_bestScore_simToRecoMerged_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster

        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToRecoMerged(self.associations))

        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )

    @cached_property
    def assocs_bestScore_simToRecoMerged_wTICL_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster

        Index eventInternal ts_id, column : caloparticle_id
        """
        # Get largest association score
        #assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToRecoMerged(self.associations))
        assocs_simToReco_largestScore = assocs_bestScore_wTICL(assocs_zip_simToRecoMerged(self.associations),TICLCandidates_zipped(self.candidates))
        #print(assocs_simToReco_largestScore)
        # Make a df out of it : index eventInternal ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )

    @cached_property
    def assocs_bestScore_simToRecoShared_df(self) -> pd.DataFrame:
        """Make a DataFrame of associations of each SimTrackster with shared energy > 0.

        Index: eventInternal, ts_id
        Columns: caloparticle_id
        """
        # Get associations where sharedE > 0
        assocs_simToReco_sharedE = assocs_zip_simToReco(self.associations)

        assocs_simToReco_sharedE = assocs_simToReco_sharedE[assocs_simToReco_sharedE["sharedE"] > 0]

        # Make a DataFrame out of it: index eventInternal, ts_id; columns: caloparticle_id
        print(assocs_simToReco_sharedE)
        def levelname(x):
            names = ["eventInternal", "caloparticle_id_wrong"] + [f"level_{i}" for i in range(x - 2)]
            return names[x] if x < len(names) else f"level_{x}"
        return (
            ak.to_dataframe(assocs_simToReco_sharedE[["ts_id", "caloparticle_id", "score", "sharedE"]],
                            levelname=levelname)
            .reset_index("caloparticle_id_wrong", drop=True)
            .set_index("caloparticle_id", append=True)
        )


    def assocs_bestScore_recoMergedToSim_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster

        Parameters :
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoMergedToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )
class Step3Reader:
    """ Reads step3 split EDM files """
    def __init__(self, file: Union[str, uproot.ReadOnlyDirectory]) -> None:
        try:
            self.eventsTree = file["Events"]
            self.file = file
        except TypeError:
            self.file = uproot.open(file)
            self.eventsTree = self.file["Events"]


class FWLiteDataframesReader:
    """ Reads pandas dataframe made by the FWLite step3 -> pandas dumper """
    def __init__(self, folder:str, sampleNb:int, readerForEventMapping:DumperReader) -> None:
        self.folder = folder
        self.sampleNb = sampleNb
        self.eventMapping = ak.to_dataframe(readerForEventMapping.tracksters.event_, levelname=lambda x:"eventInternal", anonymous="event_").reset_index()
        """ Dataframe mapping eventInternal (the index into the TICLDumper file) and the actual CMS event number (event_) """

    def readDataframe(self, name:str) -> pd.DataFrame:
        return pd.read_pickle(Path(self.folder) / f"{name}_{self.sampleNb}.pkl.gz")

    @cached_property
    def supercls(self) -> pd.DataFrame:
        return pd.merge(self.readDataframe("supercls"), self.eventMapping, on="event_").set_index(["eventInternal", "supercls_id", "ts_in_supercls_id"])

#ticlTracksters_ticlTracksterLinksSuperclustering_CLUE3D_RECO./ticlTracksters_ticlTracksterLinksSuperclustering_CLUE3D_RECO.obj/ticlTracksters_ticlTracksterLinksSuperclustering_CLUE3D_RECO.obj.regressed_energy_

