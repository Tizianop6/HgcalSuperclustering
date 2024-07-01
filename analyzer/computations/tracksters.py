from analyzer.driver.computations import DataframeComputation
from analyzer.dumperReader.reader import DumperReader, tracksters_getSeeds, tracksters_toDf, trackster_basic_fields, CPToTracksterProperties
from analyzer.driver.fileTools import SingleInputReader

# cannot use a lambda as multirprocessing does not work due to pickle issues
def _seedTracksterProperties_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return tracksters_toDf(tracksters_getSeeds(reader.tracksters_zipped[trackster_basic_fields]))
tracksters_seedProperties = DataframeComputation(_seedTracksterProperties_fct, "tracksters_seedProperties")

def _CPtoTrackster_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToReco_df, reader.tracksters_zipped[trackster_basic_fields],
            reader.simTrackstersCP_df)
CPtoTrackster_properties = DataframeComputation(_CPtoTrackster_fct, "CPtoTrackster_properties")

def _CPtoTracksterAllShared_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToRecoShared_df, reader.tracksters_zipped[trackster_basic_fields],
            reader.simTrackstersCP_df)
CPtoTracksterAllShared_properties = DataframeComputation(_CPtoTracksterAllShared_fct, "CPtoTracksterAllShared_properties")


def _CPtoTracksterMerged_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToRecoMerged_df, reader.trackstersMerged_zipped[trackster_basic_fields],
            reader.simTrackstersCP_df)
CPtoTracksterMerged_properties = DataframeComputation(_CPtoTracksterMerged_fct, "CPtoTracksterMerged_properties")