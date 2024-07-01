""" Tools to read recoToSim associations (ie trackster -> SimTrackster) from TICLDumper output """
import awkward as ak
import pandas as pd
from typing import Union

from .assocs import assocs_bestScore, assocs_zip_simToReco
from .tracksters import trackster_joinSupercluster, tracksters_groupBy, _convertTsToDataframe

def superclusterAssociatedToSimTracksterCP(supercls:ak.Array, associations:ak.Array) -> ak.Array:
    """ Returns the supercluster ids corresponding to the CaloParticle for each event (from the simTracksterCP collection) 
    Warning : Removes all CPs that are associated to a trackster not in a supercluster !
    Parameters : 
     - supercls : type: nevts * var * var * uint64 
     - associations : type nevts * {
        tsCLUE3D_simToReco_CP: var * var * uint32,
        tsCLUE3D_simToReco_CP_score: var * var * float32,
        tsCLUE3D_simToReco_CP_sharedE: var * var * float32
        ***
    }
    Returns type: nevts * 2 (nb of simTracksterCP) * int64 (supercls id)

    """
    assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToReco(associations))

    df_mainAssoc = ak.to_dataframe(assocs_simToReco_largestScore.ts_id, levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x]).reset_index(level=1)
    df_supercls = ak.to_dataframe(supercls, levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x]).reset_index(level=[1, 2])

    merged_df = pd.merge(df_mainAssoc, df_supercls,
        left_on=["eventInternal", "values"],
        right_on=["eventInternal", "values"],
        how="left" # keep the CP row event if there is no matching supercluster
    )

    # transform back to awkward array
    return ak.unflatten(ak.from_numpy(merged_df.supercls_id.to_numpy()), ak.run_lengths(ak.from_numpy(merged_df.index.to_numpy())))


def CPToTracksterProperties(assocs_bestScore_simToReco_df:pd.DataFrame, tracksters:Union[ak.Array, pd.DataFrame], simTrackstersCP_df:pd.DataFrame):
    """ For each CaloParticle, get the best associated trackster properties
    Parameters :
     - tracksters : dataframe (or zipped akward array) of tracksters with properties to keep
    """
    return (assocs_bestScore_simToReco_df
        .join(_convertTsToDataframe(tracksters), on=["eventInternal", "ts_id"])
        .join(simTrackstersCP_df, rsuffix="_CP")
    )
def CPToTracksterMergedProperties(assocs_bestScore_simToReco_df:pd.DataFrame, tracksters:Union[ak.Array, pd.DataFrame], simTrackstersCP_df:pd.DataFrame):
    """ For each CaloParticle, get the best associated trackster properties
    Parameters :
     - tracksters : dataframe (or zipped akward array) of tracksters with properties to keep
    """
    return (assocs_bestScore_simToReco_df
        .join(_convertTsToDataframe(tracksters), on=["eventInternal", "ts_id"])
        .join(simTrackstersCP_df, rsuffix="_CP")
    )

def getCPToSuperclusterProperties(supercluster_all_df:pd.DataFrame, assocs_bestScore_simToReco_df:pd.DataFrame, 
            tracksters_all:Union[pd.DataFrame, ak.Array], simTrackstersCP_df:pd.DataFrame) -> pd.DataFrame:
    """ Returns the properties of superclusters associated to each CaloParticle of the event 

    Parameters :
        - supercluster_all_df
        - assocs_bestScore_simToReco_df
        - tracksters_all : trackster data to use (all fields aggregated through tracksters_groupBy)
    
    Returns pd.DataFrame with
    Index : eventInternal, caloparticle_id (0 or 1),
    Columns :  supercls_id seed_ts_id (seed trackster id), raw_energy_supercls	regressed_energy_supercls"""
    tracksters_all = _convertTsToDataframe(tracksters_all)
    # swap index around so we have Index : eventInternal, ts_id
    df_mainAssoc = (assocs_bestScore_simToReco_df
        .reset_index("caloparticle_id")
        .set_index("ts_id", append=True)
        .join(tracksters_all.rename(columns={str(col) : str(col)+"_seed" for col in tracksters_all.columns}))
    )

    # Join best associated trackster for each CP to all other tracksters in same supercluster
    # keep caloparticle_id for later
    tsInSupercls_df = trackster_joinSupercluster(df_mainAssoc, tracksters_all, supercluster_all_df, tracksterColumnsToKeep=["caloparticle_id"]+[str(col)+"_seed" for col in tracksters_all.columns])

    # aggregate tracksters in superclusters
    supercls_grouped_df = (tsInSupercls_df
        .groupby(["eventInternal", "supercls_id"])
        .agg(**tracksters_groupBy(tsInSupercls_df, suffix="_supercls_sum"), caloparticle_id=pd.NamedAgg("caloparticle_id", "first"))
        .reset_index("supercls_id")
        .set_index("caloparticle_id", append=True)
    )

    # merge with CaloParticle information
    merged_CP = pd.merge(supercls_grouped_df, simTrackstersCP_df,
                    how="left", right_index=True, left_index=True)
    return merged_CP.rename(columns={str(col) : str(col)+"_CP" for col in simTrackstersCP_df.columns})

    # TODO technically 2 CaloParticle could be matched to the same trackster. What would happen ?


