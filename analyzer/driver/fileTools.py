from pathlib import Path
import re
import concurrent.futures
from itertools import islice
from functools import cached_property
from collections import defaultdict
import warnings
from typing import Iterable, Callable, Union, List, Dict, Tuple
from enum import Enum

import uproot
import pandas as pd
from tqdm.auto import tqdm 

from analyzer.dumperReader.reader import DumperReader, Step3Reader, FWLiteDataframesReader
from analyzer.dumperReader.dnnSampleReader import DNNSampleReader

DumperType = Enum("DumperType", ["TICL", "TICLsupercls", "SuperclsSample", "DNNStep3", "DNNDataframes"])
""" TICL : TICL dumper (does not necessarily contain superclustering tree)
TICLsupercls : TICL dumper holding superclustering tree
SuperclsSample : output of SuperclusteringSampleDumper
DNNStep3 : step3 in split mode for DNN CaloCluster+GsfElectrons (not used as problems reading event number with uproot)
DNNDataframes : output of the FWLite dumper
"""

def _dumperTypesFromFile(filePath:Path) -> list[DumperType]:
    """ Open a file and look for what kind of dumper it is """
    res = []
    with uproot.open(filePath) as file:
        fileKeys = file.keys(recursive=False, cycle=False)
        if "ticlDumper" in fileKeys:
            # TICL dumper
            ticlDumperKeys = file["ticlDumper"].keys(recursive=False, cycle=False)
            if "associations" in ticlDumperKeys:
                res.append(DumperType.TICL)
            if "superclustering" in ticlDumperKeys:
                res.append(DumperType.TICLsupercls)
        if "superclusteringSampleDumper" in fileKeys:
            res.append(DumperType.SuperclsSample)
        if "Events" in fileKeys:
            res.append(DumperType.DNNStep3)
    return res

        
class SingleInputReader:
    def __init__(self, sampleNb=None) -> None:
        self.sampleNb = sampleNb
        self._paths:dict[DumperType, Path] = dict()
        """ DumperType -> Path to file """
        self._files = dict()

    def addFile(self, path:Path, dumperTypes:list[DumperType]) -> None:
        for dumperType in dumperTypes:
            if dumperType in self._paths:
                warning = f"Duplicate {dumperType} in {self._paths[dumperType]} and {path}"
                warnings.warn(warning)
                #raise RuntimeError(f"Duplicate {dumperType} in {self._paths[dumperType]} and {path}")
            self._paths[dumperType] = path

    def _openFile(self, dumperType:DumperType):
        path = self._paths[dumperType]
        file = uproot.open(str(path))
        # for otherDumperType, otherPath in self._paths.items():
            #if path == otherPath:
            #    self._files[otherDumperType] = file
        return file

    def getFileForDumperType(self, dumperType:DumperType):
        try:
            return self._files[dumperType]
        except KeyError:
            return self._openFile(dumperType)

    @property
    def availableDumperTypes(self) -> list[DumperType]:
        return list(self._paths.keys())

    @cached_property
    def ticlDumperReader(self) -> DumperReader:
        inputFiles = []
        for dumperType in (DumperType.TICLsupercls, DumperType.TICL):
            try:
                inputFiles.append(self.getFileForDumperType(dumperType))
            except KeyError:
                pass
        return DumperReader(inputFiles)

    @cached_property
    def dnnSampleDumperReader(self) -> DNNSampleReader:
        return DNNSampleReader(self.getFileForDumperType(DumperType.SuperclsSample))
    
    @cached_property
    def step3Reader(self) -> Step3Reader:
        """ Does not really work """
        return Step3Reader(self.getFileForDumperType(DumperType.DNNStep3))
    
    @cached_property
    def nEvents(self) -> int:
        try:
            return self.ticlDumperReader.nEvents
        except:
            return self.dnnSampleDumperReader.nEvents
    
    def __repr__(self) -> str:
        res = "SingleInputReader("
        for dumperType, path in self._paths.items():
            res += f"{dumperType}={str(path)}, "
        return res + ")"

class SingleInputReaderFWLite:
    def __init__(self, basePath_ticlDumper:str, basePath_fwlite:str, sampleNb:int) -> None:
        self.basePath_ticlDumper = basePath_ticlDumper
        self.basePath_fwlite = basePath_fwlite
        self.sampleNb = sampleNb

    @cached_property  
    def ticlDumperReader(self):
        return DumperReader(str(Path(self.basePath_ticlDumper) / f"ticlDumper_mustache_{self.sampleNb}.root"))
    @cached_property
    def fwliteDataframesReader(self):
        return FWLiteDataframesReader(self.basePath_fwlite, self.sampleNb, self.ticlDumperReader)
    @cached_property
    def nEvents(self) -> int:
        return self.ticlDumperReader.nEvents
    
class Computation:
    def workOnSample(self, reader:SingleInputReader):
        """ Do work on one sample, returning a result (must be pickleable) """
        raise NotImplementedError()

    def reduce(self, results:Iterable, store:pd.HDFStore, nbOfEvents:Iterable[int]):
        """ Reduce the results given (objects returned by workOnSample) and then store them
        Parameters : 
         - nbOfEvents : list of same length as results, holding the number of event in each batch
        """
        raise NotImplementedError()



class DumperInputManager:
    def __init__(self, inputFolder: Union[str, List[str], Dict[Union[DumperType, Tuple[DumperType]], str]],
                 limitFileCount: int = None,
                 restrictToAvailableDumperTypes: List[DumperType] = None) -> None:
        """ 
        Parameters : 
            - inputFolder : can be a folder or list of folders. In case of list, for the same sampleId, the items later in the list take priority 
            - restrictToAvailableDumperTypes : ignore all samples for which we do not have all the dumperTypes specified
        """
        #pattern_dumper = re.compile(r"[a-zA-Z_\-0-9]{0,}[dD]umper_([0-9]{1,})\.root")
        pattern_dumper = re.compile(r"[a-zA-Z_\-0-9]{1,}_([0-9]{1,})\.root")
        self.inputPerSample: Dict[int, SingleInputReader] = dict()

        if isinstance(inputFolder, dict):  # dict mode
            for dumperTypesForFolder_iter, singleInputFolder in inputFolder.items():
                dumperTypesForFolder = {dumperTypesForFolder_iter} if isinstance(dumperTypesForFolder_iter, DumperType) else set(dumperTypesForFolder_iter)
                singleInputFolder = Path(singleInputFolder)
                assert singleInputFolder.is_dir(), "Input should be a folder or list of folders in folder mode"
                for child in singleInputFolder.iterdir():
                    try:
                        sampleNb = int(re.fullmatch(pattern_dumper, child.name).group(1))
                        try:
                            dumperTypes = dumperTypesForFolder.intersection(_dumperTypesFromFile(child))
                            if len(dumperTypes):
                                if sampleNb not in self.inputPerSample:
                                    self.inputPerSample[sampleNb] = SingleInputReader(sampleNb=sampleNb)
                                self.inputPerSample[sampleNb].addFile(child, dumperTypes)
                        except Exception as e:
                            print(e)
                    except AttributeError:
                        pass  # file does not match pattern
                    except Exception as e:
                        print("Exception occurred whilst reading file " + str(child) + " : " + str(e))

        else:
            # folder mode
            if isinstance(inputFolder, str):
                inputFolder = [inputFolder]

            for singleInputFolder in inputFolder:
                singleInputFolder = Path(singleInputFolder)
                assert singleInputFolder.is_dir(), "Input should be a folder or list of folders in folder mode"
                for child in singleInputFolder.iterdir():
                    try:
                        sampleNb = int(re.fullmatch(pattern_dumper, child.name).group(1))
                        try:
                            dumperTypes = _dumperTypesFromFile(child)
                            if len(dumperTypes) > 0:
                                if sampleNb not in self.inputPerSample:
                                    self.inputPerSample[sampleNb] = SingleInputReader(sampleNb=sampleNb)
                                self.inputPerSample[sampleNb].addFile(child, dumperTypes)
                        except Exception as e:
                            print(e)
                    except AttributeError:
                        pass  # file does not match pattern
                    except Exception as e:
                        print("Exception occurred whilst reading file " + str(child) + " : " + str(e))

            if restrictToAvailableDumperTypes is not None:
                self.restrictToAvailableTypes(restrictToAvailableDumperTypes)

        if limitFileCount is not None:
            self.inputPerSample = dict(islice(self.inputPerSample.items(), limitFileCount))
        
    @property
    def inputCount(self):
        return len(self.inputPerSample)
        
    @property
    def inputReaders(self) -> list[SingleInputReader]:
        return list(self.inputPerSample.values())
    
    def restrictToAvailableTypes(self, dumperTypes:list[DumperType]):
        """ Remove all samples for which we do not have all the dumperTypes """
        dumperTypes = set(dumperTypes)
        samplesToRemove = [sampleId for sampleId, singleInput in self.inputPerSample.items() if not dumperTypes.issubset(singleInput.availableDumperTypes)]
        for key in samplesToRemove:
            del self.inputPerSample[key]


def _map_fcn(input:SingleInputReader, computations:list[Computation]) -> list:
    try:
        return [comp.workOnSample(input) for comp in computations], input.nEvents
    except uproot.DeserializationError as e:
        raise RuntimeError("uproot.DeserializationError was raised in worker process whilst processing file " + str(input.sampleNb) + "\nThe message was " + str(e))


from typing import Union

def runComputations(computations:list[Computation], inputManager:Union[DumperInputManager, list[SingleInputReader]], store:Union[pd.HDFStore, None]=None, max_workers=10):
    """ Run the list of computations given, eventually in parallel
    
    Parameters : 
     - max_workers : if >1, run in multiprocessing mode with that many worker processes (if 1, run serially)
    """
    if isinstance(inputManager, DumperInputManager):
        inputReader = inputManager.inputReaders
    else:
        inputReader = inputManager
    if max_workers != 1:
        def wrapper(gen):
            while True:
                try:
                    yield next(gen)
                except StopIteration:
                    break
                except Exception as e:
                    print(e) # or whatever kind of logging you want
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(max_workers, len(inputReader))) as executor:
            # Deal with exceptions in handling of samples without failing everything
            # map_iterator is a generator that will yield the results in order, waiting until they are ready if necessary
            map_iterator = iter(tqdm(executor.map(_map_fcn, inputReader, [computations]*len(inputReader)), total=len(inputReader)))
            # We need this loop to carry on building map_res list even if there is an exception
            map_res = list()
            while True:
                try:
                    map_res.append(next(map_iterator))
                except StopIteration:
                    break # exhausted all samples
                except Exception as e:
                    print("An exception occured during processing of a sample. Exception details are : ")
                    print(e)
            
    else:
        map_res = map(_map_fcn, inputReader, [computations]*len(inputReader))
    
    # map_res is [([comp1Res, comp2Res, ...], nEvts), ...]
    resultsPerFile, nbOfEvents = zip(*map_res)
    # listOfResults is ([comp1Res, comp2Res, ...], ...)

    # reduce
    compResults = []
    for comp, resultsPerComputation in zip(computations, zip(*resultsPerFile)):
        # resultsPerComputation is a tuple of ResultToReduce that are to be reduced together
        compResults.append(comp.reduce(resultsPerComputation, store, nbOfEvents=nbOfEvents))
    return compResults
