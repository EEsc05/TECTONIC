from ..common import *
BASE=ROOT;ROOT=BASE/'historical_v3';VERSION='historical_v3';SEED=20261105
for part in ['raw','manifests','splits','processed','models','reports','logs','figures','tests']:(ROOT/part).mkdir(exist_ok=True,parents=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
NAMES={17:'17_historical_data_expansion',18:'18_historical_rolling_windows',19:'19_stage1_deterministic_expansion',20:'20_stage1_generative',21:'21_stage2_seismic_v3',22:'22_stage2_gas',23:'23_stage2_ash',24:'24_stage2_explosion_expansion',25:'25_multioutput_feasibility',26:'26_cross_stage_probabilistic',27:'27_integrated_evaluation'}
def experiment(i):
 p=BASE/'experiments'/NAMES[i]/'runs'/VERSION;p.mkdir(parents=True,exist_ok=True);return p
SITES={'Asosan':'aso','Kirishimayama':'kirishima','Sakurajima':'sakurajima','Hakoneyama':'hakone','Miyakejima':'miyakejima','Azumayama':'azumayama','Asamayama':'asamayama','Unzendake':'unzendake'}
QUARANTINE={'Fujisan':'previous_original_geographic_test_system','Iwatesan':'previous_v2_geographic_evaluation_system','Tokachidake':'previous_v2_geographic_evaluation_system'}
GEO=['asamayama','unzendake']
CONFIG=dict(dataset_version=VERSION,evaluation_version='historical_eval_v3',seed=SEED,mode='retrospective',horizon_days=[1,60],lookbacks=[30,60,90,180,365],quarantine_sites=list(QUARANTINE),quarantine_period='all inherited Japanese2022+records remain reference/quarantined from new primary historical model selection',geographic_evaluation_sites=GEO,protocol='Separate modality-specific chronology will be preregistered after source coverage audit and before model fitting; no old frozen result is relabelled',training_origin_strides=[1,7,30],evaluation_origin_stride=61,activity_threshold='Future mean rate exceeds2times past90dayrate and increases by>=1event/day; extreme daily count exceeds max(past90p95, past90mean+3sqrt(past90mean+1)); thresholds fixed without test outcomes',neural_advance=dict(minimum_independent_training_blocks=50,minimum_training_volcanoes=4,maximum_extreme_reconstruction_fraction=.25,minimum_validation_crps_improvement_fraction=.05,coverage90_range=[.80,.98]),spatial_grid='32native pixels; no geometry/reference stitching; differences in physical footprint retained',max_training_neural_epochs=100,samples_per_forecast=32)

def protect():
 p=ROOT/'manifests/frozen_reference_hashes.json'
 if p.exists():return readjson(p)
 paths=[]
 for directory in [BASE/'models',BASE/'data',BASE/'results',BASE/'stage2_expansion_v1']:
  for file in directory.rglob('*'):
   if file.is_file() and file.suffix not in ['.jsonl'] and 'historical_v3' not in file.parts:paths.append(file)
 # File content hashes preserve all prior numerical results/checkpoints/splits, not only filenames.
 hashes={str(f.relative_to(REPO)):digest(f) for f in paths};writejson(p,hashes);return hashes

def verify_frozen():
 hashes=readjson(ROOT/'manifests/frozen_reference_hashes.json');bad=[name for name,h in hashes.items() if digest(REPO/name)!=h];assert not bad,bad[:5];return len(hashes)

def register_protocol(extra):
 p=ROOT/'splits/preregistered_protocol.json';new={**CONFIG,**extra}
 if p.exists():assert readjson(p)==new,'Protocol is immutable; start a new evaluation version to change it'
 else:writejson(p,new);writejson(ROOT/'manifests/protocol_registration.json',dict(registered_utc=utc(),sha256=digest(p),fitted_models_present=bool(list((ROOT/'models').glob('*')))))
 return new
