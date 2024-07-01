""" Tools for doing dataframes of tracksters from TICLDumper output """
import awkward as ak
import pandas as pd
import numpy as np
from typing import Union

from .assocs import assocs_toDf

trackster_basic_fields = ["ts_id", "raw_energy", "raw_em_energy", "regressed_energy", "raw_pt", "raw_em_pt", "barycenter_eta", "barycenter_phi", 'barycenter_x', 'barycenter_y']

def tracksters_toDf(tracksters:ak.Array) -> pd.DataFrame:
    """ Makes a dataframe with all tracksters
    Index : eventInternal, ts_id
    """
    assert tracksters.ndim == 2, "Tracksters ak.Array should not include vertices information (or other nested fields)"
    try:
        if "ts_id" in tracksters.fields:
            return (ak.to_dataframe(tracksters, 
                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
            )
        else:
            return ak.to_dataframe(tracksters, 
                levelname=lambda x : {0:"eventInternal", 1:"ts_id"}[x])
    except KeyError as e:
        if e.args[0] == 2:
            raise ValueError("Tracksters ak.Array should not include vertices information (or other nested fields)") from e
        else:
            raise e
            

def _convertTsToDataframe(tracksters:Union[ak.Array,pd.DataFrame]) -> pd.DataFrame:
    """ COnvert if needed an ak.Array of tracksters to dataframe """
    if isinstance(tracksters, ak.Array):
        return tracksters_toDf(tracksters)
    else:
        return tracksters

def tracksters_joinWithSimTracksters(tracksters:ak.Array, simTracksters:ak.Array, assoc:ak.Array, score_threshold=assocs_toDf.__defaults__[0]) -> pd.DataFrame:
    """ Make a merged dataframe holding trackster information joined with the sim tracksters information.
    Only tracksters with an association are kept.
    Index : eventInternal ts_id
    """
    df_merged_1 = pd.merge(
        tracksters_toDf(tracksters),
        assocs_toDf(assoc, score_threshold=score_threshold),
        left_index=True, right_index=True
    )
    return pd.merge(
        df_merged_1,
        tracksters_toDf(simTracksters),
        left_on=["eventInternal", "simts_id"], right_index=True,
        how="left",
        suffixes=(None, "_sim")
    )


def tracksters_getSeeds(tracksters_zipped:ak.Array) -> ak.Array:
    """ For each event, select the seed tracksters (largest pt trackster for each endcap) 
    
    From : nevts * nTs * { trackster props} 
    To : nevts * 2 (possibly 1 or 0) * {trackster props}
    """
    if "raw_pt" not in tracksters_zipped.fields or "ts_id" not in tracksters_zipped.fields:
        raise ValueError("Need raw_pt and ts_id in tracksters array")
    # Sort by decreasing pT
    # Using ak.sort destroys the records (dangerous ! )
    #sorted_ar = ak.sort(tracksters_zipped, ascending=False,
    #    behavior={(np.less_equal, "trackster", "trackster",) : lambda x, y: x.raw_pt <= y.raw_pt})
    # so use argsort instead
    sort_idx = ak.argsort(tracksters_zipped.raw_pt, ascending=False)
    sorted_ar = tracksters_zipped[sort_idx]

    # ak.singletons(ak.firsts(...)) takes the first element of the list then outputs a list with one element
    # in case of no elements in list, does not crash but make an empty list
    # Then we concat the lists together
    return ak.concatenate([ak.singletons(ak.firsts(sorted_ar[sorted_ar.barycenter_eta <0])), ak.singletons(ak.firsts(sorted_ar[sorted_ar.barycenter_eta>0]))],
        axis=1)


def supercluster_joinTracksters(supercluster_df:pd.DataFrame, tracksters:Union[ak.Array, pd.DataFrame]) -> pd.DataFrame:
    """ Add trackster information to supercluster dataframe """
    tracksters = _convertTsToDataframe(tracksters)
    return supercluster_df.join(tracksters, on=["eventInternal", "ts_id"])

def trackster_joinSupercluster(tracksters: Union[pd.DataFrame, ak.Array], allTracksters: Union[pd.DataFrame, ak.Array],
            supercluster_df: pd.DataFrame, tracksterColumnsToKeep: list[str] = []) -> pd.DataFrame:
    """ From a trackster, add all the other tracksters in the supercluster 
    
    Tracksters not in a supercluster will be dropped.
    Parameters : 
     - tracksters: a Dataframe with index eventInternal, ts_id (columns are ignored)
     - allTracksters : a Dataframe holding information on tracksters wanted in output (tracksters is subset of allTracksters)
     - supercluster_df : superclusters
     - tracksterColumnsToKeep : columns to keep in output from "tracksters", that are not in "allTracksters" 
    """
    tracksters = _convertTsToDataframe(tracksters)
    allTracksters = _convertTsToDataframe(allTracksters)
    # merge with superclusters to get supercls_id
    merged_step1 = pd.merge(
        tracksters[tracksterColumnsToKeep],  # we don't want any of the columns here 
        # use convert_dtypes to avoid converting ids to float when there are missing values
        supercluster_df.reset_index(["supercls_id", "ts_in_supercls_id"]).set_index("ts_id", append=True).convert_dtypes(),
        left_index=True, right_index=True,
        how="inner"  # drops tracksters not in supercls
    )
    # merge again to get all the other tracksters in supercluster
    merged_step2 = (pd.merge(
            merged_step1,
            supercluster_df,
            left_on=["eventInternal", "supercls_id"],
            right_on=["eventInternal", "supercls_id"],
            how="inner"
        )
        .set_index("ts_id", append=True)
    )

    # final merge with allTracksters to get trackster properties back
    return pd.merge(
        merged_step2,
        allTracksters,
        left_index=True, right_index=True,
        how="inner",
    )


def tracksters_groupBy(tracksters_df:pd.DataFrame, suffix="_sum"):
    """ Helper for trackster groupby. Use as tracksters_df.groupby(["eventInternal", "supercls_id"]).agg(**tracksters_groupBy(tracksters_df)) """
    fields = tracksters_df.columns
    sum_fields = ["raw_energy", "raw_em_energy", "regressed_energy"]
    res = dict()
    for field in fields:
        if field in sum_fields: 
            res[field+suffix] = pd.NamedAgg(field, "sum")
        elif field.endswith("_seed"):
            res[field] = pd.NamedAgg(field, "first")

    return res