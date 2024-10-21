# Modified in order to add a script with the aim study the time resolution of HGCAL and MTD at 0 PU
Script is /analyzer/PerformanceTICLv5/TimeRes_PU0.py.

Also some scripts regarding track pT resolution are present in the same folder. 

# Electron superclustering studies in HGCAL
Based off Alessandro Tarabini's work

There is a conda environment : supercls-analyzer-env.yml 

## Organization
 - analyzer : python code for making performance plots
 - cmssw_jobs : archive of all condor jobs submitted (submit files + logs)
 - dnn_training : retraining of the DNN (fairly self-contained)
 - lawJobs : ignore
 - runs : store of all CMSSW python config files used

## Analyzer
### Organization
 - dumperReader : code to read TICLDumper output and study tracksters, superclusters and sim associations
 - energy_resolution : making energy resolution plots
 - computations & driver : make dataframes at scale and in parallel for resolution plots with lots of statistics. 
 - egamma : looking at GSFElectrons, mostly from CMSSW FEVTDEBUG files
 - 
 
## Instructions for using DQM to make plots
### Making step3
You need a CMSSW config for step3 that runs VALIDATION & DQM (with phase2 modifiers) as well as datatier & eventcontent for DQM. For example : `cmsDriver.py step3 -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --datatier DQMIO --eventcontent DQM ...` (one can also put FEVTDEBUGHLT in eventcontent if wanted). RunTheMatrix workflows normally include this. Probably things could be removed (miniAOD validation is probably unnecessary). This will create an output file called `step3_inDQM.root` (or `step3.root` if FEVT was not requested). It is not readable as such.

Example config running full validation+DQM : (runs/NewElectronsD98_PRtests/bc18-ticlMustache/step3.py)

### step4 : harvesting
The `step3_inDQM.root` files are not readable : need step4 (HARVESTING).
cmsDriver example : `cmsDriver.py step4 -s HARVESTING:@phase2Validation+@phase2+@miniAODValidation+@miniAODDQM  --filetype DQM --filein file:step3_inDQM.root --fileout file:step4.root ...`
If running multiple step3 jobs, one needs to run only one step4 to merge everything. The input files should be updated in the config, for example : 
~~~python
# Input source
import glob
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(['file:' + x for x in glob.glob("step3_inDQM*.root")])
    #fileNames = cms.untracked.vstring([f'file:step3_inDQM_{i}.root' for i in range(1, 31)])
)
~~~
See also a concrete example in (runs/NewElectronsD98_PRtests/bc18-ticlMustache/step4.py).

This outputs a file `DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root` containing a lot of histograms. Intersesting ones are : 
#### EgammaV
Egamma validation plots. In [ElectronMcSignalValidator](https://github.com/cms-sw/cmssw/blob/master/Validation/RecoEgamma/plugins/ElectronMcSignalValidator.cc) are electron histograms made from GsfElectrons. Using genparticle matching (deltaR based I think). Efficiency, energy/genEnergy plots etc.

#### HGCAL
HGCAL validation. In [HGCalValidator](https://github.com/cms-sw/cmssw/blob/master/Validation/HGCalValidation/plugins/HGCalValidator.cc) there are many plots for trackster validation.
