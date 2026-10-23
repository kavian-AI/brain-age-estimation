# OpenNeuro healthy adults through FastSurfer + SynthStrip (GPU), streamed from S3.
#
# 12 CC0 OpenNeuro datasets (Precision Aging Network, NIMH Healthy Volunteers, MiND,
# Neurocognitive aging, GRAND, adult-lifespan sets, ...): 2,618 healthy people, 1,356
# aged 60+, plus 667 extra scans (second sessions / same-session runs) for repeat noise.
# Per scan: FastSurfer v2.4.2 aseg+DKT volumes (same program as IXI / DLBS / OASIS-2),
# SynthStrip intracranial volume (head size), and the scanner from the BIDS sidecar
# (manufacturer, model, field strength) so each scanner can be its own "site".
# Chunks of 12 scans: the next chunk downloads while the GPU works; rows are appended
# after every chunk, so a timeout loses at most one chunk.
import concurrent.futures as cf
import glob, gzip, json, os, shutil, subprocess, threading, time, urllib.request

PART = 159
T0 = time.time()
BUDGET_S = 9.4 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["oasis1__OAS1_0266_MR1__mpr-2", "oasis1:disc7:OAS1_0266_MR1:mpr-2"], ["ds007522__sub-HML0179__3088", "ds007522/sub-HML0179/ses-follow01/anat/sub-HML0179_ses-follow01_rec-defaced_T1w.nii.gz"], ["ds005405__sub-079__3089", "ds005405/sub-079/ses-bhb/anat/sub-079_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-079__3090", "ds005405/sub-079/ses-glc/anat/sub-079_ses-glc_run-1_T1w.nii"], ["ds005405__sub-079__3091", "ds005405/sub-079/ses-glc/anat/sub-079_ses-glc_run-2_T1w.nii"], ["ds005405__sub-139__3092", "ds005405/sub-139/ses-bhb/anat/sub-139_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-139__3093", "ds005405/sub-139/ses-glc/anat/sub-139_ses-glc_run-1_T1w.nii"], ["ds005405__sub-139__3094", "ds005405/sub-139/ses-glc/anat/sub-139_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0168_MR1__mpr-2", "oasis1:disc5:OAS1_0168_MR1:mpr-2"], ["oasis1__OAS1_0275_MR1__mpr-2", "oasis1:disc8:OAS1_0275_MR1:mpr-2"], ["oasis1__OAS1_0323_MR1__mpr-2", "oasis1:disc9:OAS1_0323_MR1:mpr-2"], ["oasis1__OAS1_0327_MR1__mpr-2", "oasis1:disc9:OAS1_0327_MR1:mpr-2"], ["oasis1__OAS1_0434_MR1__mpr-2", "oasis1:disc12:OAS1_0434_MR1:mpr-2"], ["oasis1__OAS1_0071_MR1__mpr-2", "oasis1:disc2:OAS1_0071_MR1:mpr-2"], ["oasis1__OAS1_0181_MR1__mpr-2", "oasis1:disc5:OAS1_0181_MR1:mpr-2"], ["oasis1__OAS1_0222_MR1__mpr-2", "oasis1:disc6:OAS1_0222_MR1:mpr-2"], ["ds005405__sub-054__3095", "ds005405/sub-054/ses-bhb/anat/sub-054_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-054__3096", "ds005405/sub-054/ses-glc/anat/sub-054_ses-glc_run-1_T1w.nii"], ["ds005405__sub-054__3097", "ds005405/sub-054/ses-glc/anat/sub-054_ses-glc_run-2_T1w.nii"], ["ds005405__sub-091__3098", "ds005405/sub-091/ses-bhb/anat/sub-091_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-091__3099", "ds005405/sub-091/ses-glc/anat/sub-091_ses-glc_run-1_T1w.nii"], ["ds005405__sub-091__3100", "ds005405/sub-091/ses-glc/anat/sub-091_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0020_MR1__mpr-2", "oasis1:disc1:OAS1_0020_MR1:mpr-2"], ["oasis1__OAS1_0050_MR1__mpr-2", "oasis1:disc2:OAS1_0050_MR1:mpr-2"], ["oasis1__OAS1_0063_MR1__mpr-2", "oasis1:disc2:OAS1_0063_MR1:mpr-2"], ["oasis1__OAS1_0182_MR1__mpr-2", "oasis1:disc5:OAS1_0182_MR1:mpr-2"], ["oasis1__OAS1_0188_MR1__mpr-2", "oasis1:disc5:OAS1_0188_MR1:mpr-2"], ["oasis1__OAS1_0204_MR1__mpr-2", "oasis1:disc6:OAS1_0204_MR1:mpr-2"], ["oasis1__OAS1_0213_MR1__mpr-2", "oasis1:disc6:OAS1_0213_MR1:mpr-2"], ["oasis1__OAS1_0305_MR1__mpr-2", "oasis1:disc8:OAS1_0305_MR1:mpr-2"], ["ds005405__sub-116__3101", "ds005405/sub-116/ses-bhb/anat/sub-116_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-116__3102", "ds005405/sub-116/ses-glc/anat/sub-116_ses-glc_run-1_T1w.nii"], ["ds005405__sub-116__3103", "ds005405/sub-116/ses-glc/anat/sub-116_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0044_MR1__mpr-2", "oasis1:disc2:OAS1_0044_MR1:mpr-2"], ["oasis1__OAS1_0086_MR1__mpr-2", "oasis1:disc3:OAS1_0086_MR1:mpr-2"], ["oasis1__OAS1_0096_MR1__mpr-2", "oasis1:disc3:OAS1_0096_MR1:mpr-2"], ["ds005405__sub-100__3104", "ds005405/sub-100/ses-bhb/anat/sub-100_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-100__3105", "ds005405/sub-100/ses-glc/anat/sub-100_ses-glc_run-1_T1w.nii"], ["ds005405__sub-100__3106", "ds005405/sub-100/ses-glc/anat/sub-100_ses-glc_run-2_T1w.nii"], ["ds005405__sub-104__3107", "ds005405/sub-104/ses-bhb/anat/sub-104_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-104__3108", "ds005405/sub-104/ses-glc/anat/sub-104_ses-glc_run-1_T1w.nii"], ["ds005405__sub-104__3109", "ds005405/sub-104/ses-glc/anat/sub-104_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0058_MR1__mpr-2", "oasis1:disc2:OAS1_0058_MR1:mpr-2"], ["oasis1__OAS1_0262_MR1__mpr-2", "oasis1:disc7:OAS1_0262_MR1:mpr-2"], ["oasis1__OAS1_0367_MR1__mpr-2", "oasis1:disc10:OAS1_0367_MR1:mpr-2"], ["oasis1__OAS1_0375_MR1__mpr-2", "oasis1:disc10:OAS1_0375_MR1:mpr-2"], ["ds005405__sub-053__3110", "ds005405/sub-053/ses-bhb/anat/sub-053_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-053__3111", "ds005405/sub-053/ses-glc/anat/sub-053_ses-glc_run-1_T1w.nii"], ["ds005405__sub-053__3112", "ds005405/sub-053/ses-glc/anat/sub-053_ses-glc_run-2_T1w.nii"], ["ds005405__sub-063__3113", "ds005405/sub-063/ses-bhb/anat/sub-063_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-063__3114", "ds005405/sub-063/ses-glc/anat/sub-063_ses-glc_run-1_T1w.nii"], ["ds005405__sub-063__3115", "ds005405/sub-063/ses-glc/anat/sub-063_ses-glc_run-2_T1w.nii"], ["ds005405__sub-135__3116", "ds005405/sub-135/ses-bhb/anat/sub-135_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-135__3117", "ds005405/sub-135/ses-glc/anat/sub-135_ses-glc_run-1_T1w.nii"], ["ds005405__sub-135__3118", "ds005405/sub-135/ses-glc/anat/sub-135_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0282_MR1__mpr-2", "oasis1:disc8:OAS1_0282_MR1:mpr-2"], ["oasis1__OAS1_0366_MR1__mpr-2", "oasis1:disc10:OAS1_0366_MR1:mpr-2"], ["oasis1__OAS1_0429_MR1__mpr-2", "oasis1:disc12:OAS1_0429_MR1:mpr-2"], ["ds005405__sub-101__3119", "ds005405/sub-101/ses-bhb/anat/sub-101_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-101__3120", "ds005405/sub-101/ses-glc/anat/sub-101_ses-glc_run-1_T1w.nii"], ["ds005405__sub-101__3121", "ds005405/sub-101/ses-glc/anat/sub-101_ses-glc_run-2_T1w.nii"], ["ds005405__sub-140__3122", "ds005405/sub-140/ses-bhb/anat/sub-140_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-140__3123", "ds005405/sub-140/ses-glc/anat/sub-140_ses-glc_run-1_T1w.nii"], ["ds005405__sub-140__3124", "ds005405/sub-140/ses-glc/anat/sub-140_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0178_MR1__mpr-2", "oasis1:disc5:OAS1_0178_MR1:mpr-2"], ["oasis1__OAS1_0183_MR1__mpr-2", "oasis1:disc5:OAS1_0183_MR1:mpr-2"], ["ds005405__sub-067__3125", "ds005405/sub-067/ses-bhb/anat/sub-067_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-067__3126", "ds005405/sub-067/ses-glc/anat/sub-067_ses-glc_run-1_T1w.nii"], ["ds005405__sub-067__3127", "ds005405/sub-067/ses-glc/anat/sub-067_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0027_MR1__mpr-2", "oasis1:disc1:OAS1_0027_MR1:mpr-2"], ["oasis1__OAS1_0074_MR1__mpr-2", "oasis1:disc2:OAS1_0074_MR1:mpr-2"], ["oasis1__OAS1_0190_MR1__mpr-2", "oasis1:disc5:OAS1_0190_MR1:mpr-2"], ["oasis1__OAS1_0349_MR1__mpr-2", "oasis1:disc10:OAS1_0349_MR1:mpr-2"], ["ds005405__sub-119__3128", "ds005405/sub-119/ses-bhb/anat/sub-119_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-119__3129", "ds005405/sub-119/ses-glc/anat/sub-119_ses-glc_run-1_T1w.nii"], ["ds005405__sub-119__3130", "ds005405/sub-119/ses-glc/anat/sub-119_ses-glc_run-2_T1w.nii"], ["ds005405__sub-124__3131", "ds005405/sub-124/ses-bhb/anat/sub-124_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-124__3132", "ds005405/sub-124/ses-glc/anat/sub-124_ses-glc_run-1_T1w.nii"], ["ds005405__sub-124__3133", "ds005405/sub-124/ses-glc/anat/sub-124_ses-glc_run-2_T1w.nii"], ["ds005405__sub-051__3134", "ds005405/sub-051/ses-bhb/anat/sub-051_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-051__3135", "ds005405/sub-051/ses-glc/anat/sub-051_ses-glc_run-1_T1w.nii"], ["ds005405__sub-051__3136", "ds005405/sub-051/ses-glc/anat/sub-051_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0167_MR1__mpr-2", "oasis1:disc5:OAS1_0167_MR1:mpr-2"], ["oasis1__OAS1_0336_MR1__mpr-2", "oasis1:disc9:OAS1_0336_MR1:mpr-2"], ["ds005405__sub-057__3137", "ds005405/sub-057/ses-bhb/anat/sub-057_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-057__3138", "ds005405/sub-057/ses-glc/anat/sub-057_ses-glc_run-1_T1w.nii"], ["ds005405__sub-057__3139", "ds005405/sub-057/ses-glc/anat/sub-057_ses-glc_run-2_T1w.nii"], ["ds005405__sub-114__3140", "ds005405/sub-114/ses-bhb/anat/sub-114_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-114__3141", "ds005405/sub-114/ses-glc/anat/sub-114_ses-glc_run-1_T1w.nii"], ["ds005405__sub-114__3142", "ds005405/sub-114/ses-glc/anat/sub-114_ses-glc_run-2_T1w.nii"], ["ds005405__sub-115__3143", "ds005405/sub-115/ses-bhb/anat/sub-115_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-115__3144", "ds005405/sub-115/ses-glc/anat/sub-115_ses-glc_run-1_T1w.nii"], ["ds005405__sub-115__3145", "ds005405/sub-115/ses-glc/anat/sub-115_ses-glc_run-2_T1w.nii"], ["ds005405__sub-120__3146", "ds005405/sub-120/ses-bhb/anat/sub-120_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-120__3147", "ds005405/sub-120/ses-glc/anat/sub-120_ses-glc_run-1_T1w.nii"], ["ds005405__sub-120__3148", "ds005405/sub-120/ses-glc/anat/sub-120_ses-glc_run-2_T1w.nii"], ["ds005405__sub-126__3149", "ds005405/sub-126/ses-bhb/anat/sub-126_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-126__3150", "ds005405/sub-126/ses-glc/anat/sub-126_ses-glc_run-1_T1w.nii"], ["ds005405__sub-126__3151", "ds005405/sub-126/ses-glc/anat/sub-126_ses-glc_run-2_T1w.nii"], ["ds005405__sub-141__3152", "ds005405/sub-141/ses-bhb/anat/sub-141_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-141__3153", "ds005405/sub-141/ses-glc/anat/sub-141_ses-glc_run-1_T1w.nii"], ["ds005405__sub-141__3154", "ds005405/sub-141/ses-glc/anat/sub-141_ses-glc_run-2_T1w.nii"], ["ds005405__sub-149__3155", "ds005405/sub-149/ses-bhb/anat/sub-149_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-149__3156", "ds005405/sub-149/ses-glc/anat/sub-149_ses-glc_run-1_T1w.nii"], ["ds005405__sub-149__3157", "ds005405/sub-149/ses-glc/anat/sub-149_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0088_MR1__mpr-2", "oasis1:disc3:OAS1_0088_MR1:mpr-2"], ["oasis1__OAS1_0159_MR1__mpr-2", "oasis1:disc5:OAS1_0159_MR1:mpr-2"], ["oasis1__OAS1_0018_MR1__mpr-2", "oasis1:disc1:OAS1_0018_MR1:mpr-2"], ["oasis1__OAS1_0040_MR1__mpr-2", "oasis1:disc1:OAS1_0040_MR1:mpr-2"], ["oasis1__OAS1_0384_MR1__mpr-2", "oasis1:disc11:OAS1_0384_MR1:mpr-2"], ["ds005405__sub-047__3158", "ds005405/sub-047/ses-bhb/anat/sub-047_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-047__3159", "ds005405/sub-047/ses-glc/anat/sub-047_ses-glc_run-1_T1w.nii"], ["ds005405__sub-047__3160", "ds005405/sub-047/ses-glc/anat/sub-047_ses-glc_run-2_T1w.nii"], ["ds005405__sub-106__3161", "ds005405/sub-106/ses-bhb/anat/sub-106_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-106__3162", "ds005405/sub-106/ses-glc/anat/sub-106_ses-glc_run-1_T1w.nii"], ["ds005405__sub-106__3163", "ds005405/sub-106/ses-glc/anat/sub-106_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0235_MR1__mpr-2", "oasis1:disc7:OAS1_0235_MR1:mpr-2"], ["ds005405__sub-111__3164", "ds005405/sub-111/ses-bhb/anat/sub-111_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-111__3165", "ds005405/sub-111/ses-glc/anat/sub-111_ses-glc_run-1_T1w.nii"], ["ds005405__sub-111__3166", "ds005405/sub-111/ses-glc/anat/sub-111_ses-glc_run-2_T1w.nii"], ["ds005405__sub-006__3167", "ds005405/sub-006/ses-bhb/anat/sub-006_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-006__3168", "ds005405/sub-006/ses-glc/anat/sub-006_ses-glc_run-1_T1w.nii"], ["ds005405__sub-006__3169", "ds005405/sub-006/ses-glc/anat/sub-006_ses-glc_run-2_T1w.nii"], ["ds005405__sub-013__3170", "ds005405/sub-013/ses-bhb/anat/sub-013_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-013__3171", "ds005405/sub-013/ses-glc/anat/sub-013_ses-glc_run-1_T1w.nii"], ["ds005405__sub-013__3172", "ds005405/sub-013/ses-glc/anat/sub-013_ses-glc_run-2_T1w.nii"], ["ds005405__sub-015__3173", "ds005405/sub-015/ses-bhb/anat/sub-015_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-015__3174", "ds005405/sub-015/ses-glc/anat/sub-015_ses-glc_run-1_T1w.nii"], ["ds005405__sub-015__3175", "ds005405/sub-015/ses-glc/anat/sub-015_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0173_MR1__mpr-2", "oasis1:disc5:OAS1_0173_MR1:mpr-2"], ["ds005405__sub-012__3176", "ds005405/sub-012/ses-bhb/anat/sub-012_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-012__3177", "ds005405/sub-012/ses-glc/anat/sub-012_ses-glc_run-1_T1w.nii"], ["ds005405__sub-012__3178", "ds005405/sub-012/ses-glc/anat/sub-012_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0145_MR1__mpr-2", "oasis1:disc4:OAS1_0145_MR1:mpr-2"], ["oasis1__OAS1_0409_MR1__mpr-2", "oasis1:disc11:OAS1_0409_MR1:mpr-2"], ["oasis1__OAS1_0145_MR2__mpr-1", "oasis1:disc4:OAS1_0145_MR2:mpr-1"], ["oasis1__OAS1_0145_MR2__mpr-2", "oasis1:disc4:OAS1_0145_MR2:mpr-2"], ["ds005405__sub-127__3179", "ds005405/sub-127/ses-bhb/anat/sub-127_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-127__3180", "ds005405/sub-127/ses-glc/anat/sub-127_ses-glc_run-1_T1w.nii"], ["ds005405__sub-127__3181", "ds005405/sub-127/ses-glc/anat/sub-127_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0069_MR1__mpr-2", "oasis1:disc2:OAS1_0069_MR1:mpr-2"], ["oasis1__OAS1_0318_MR1__mpr-2", "oasis1:disc9:OAS1_0318_MR1:mpr-2"], ["ds005405__sub-150__3182", "ds005405/sub-150/ses-bhb/anat/sub-150_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-150__3183", "ds005405/sub-150/ses-glc/anat/sub-150_ses-glc_run-1_T1w.nii"], ["ds005405__sub-150__3184", "ds005405/sub-150/ses-glc/anat/sub-150_ses-glc_run-2_T1w.nii"], ["ds005405__sub-058__3185", "ds005405/sub-058/ses-bhb/anat/sub-058_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-058__3186", "ds005405/sub-058/ses-glc/anat/sub-058_ses-glc_run-1_T1w.nii"], ["ds005405__sub-058__3187", "ds005405/sub-058/ses-glc/anat/sub-058_ses-glc_run-2_T1w.nii"], ["ds005405__sub-105__3188", "ds005405/sub-105/ses-bhb/anat/sub-105_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-105__3189", "ds005405/sub-105/ses-glc/anat/sub-105_ses-glc_run-1_T1w.nii"], ["ds005405__sub-105__3190", "ds005405/sub-105/ses-glc/anat/sub-105_ses-glc_run-2_T1w.nii"], ["ds005405__sub-122__3191", "ds005405/sub-122/ses-bhb/anat/sub-122_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-122__3192", "ds005405/sub-122/ses-glc/anat/sub-122_ses-glc_run-1_T1w.nii"], ["ds005405__sub-122__3193", "ds005405/sub-122/ses-glc/anat/sub-122_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0192_MR1__mpr-2", "oasis1:disc6:OAS1_0192_MR1:mpr-2"], ["oasis1__OAS1_0319_MR1__mpr-2", "oasis1:disc9:OAS1_0319_MR1:mpr-2"], ["oasis1__OAS1_0376_MR1__mpr-2", "oasis1:disc10:OAS1_0376_MR1:mpr-2"], ["ds005405__sub-085__3194", "ds005405/sub-085/ses-bhb/anat/sub-085_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-085__3195", "ds005405/sub-085/ses-glc/anat/sub-085_ses-glc_run-1_T1w.nii"], ["ds005405__sub-085__3196", "ds005405/sub-085/ses-glc/anat/sub-085_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0012_MR1__mpr-2", "oasis1:disc1:OAS1_0012_MR1:mpr-2"], ["oasis1__OAS1_0127_MR1__mpr-2", "oasis1:disc4:OAS1_0127_MR1:mpr-2"], ["oasis1__OAS1_0417_MR1__mpr-2", "oasis1:disc11:OAS1_0417_MR1:mpr-2"], ["oasis1__OAS1_0444_MR1__mpr-2", "oasis1:disc12:OAS1_0444_MR1:mpr-2"], ["ds005405__sub-094__3197", "ds005405/sub-094/ses-bhb/anat/sub-094_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-094__3198", "ds005405/sub-094/ses-glc/anat/sub-094_ses-glc_run-1_T1w.nii"], ["ds005405__sub-094__3199", "ds005405/sub-094/ses-glc/anat/sub-094_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0045_MR1__mpr-2", "oasis1:disc2:OAS1_0045_MR1:mpr-2"], ["oasis1__OAS1_0101_MR1__mpr-2", "oasis1:disc3:OAS1_0101_MR1:mpr-2"], ["oasis1__OAS1_0239_MR1__mpr-2", "oasis1:disc7:OAS1_0239_MR1:mpr-2"], ["oasis1__OAS1_0101_MR2__mpr-1", "oasis1:disc3:OAS1_0101_MR2:mpr-1"], ["oasis1__OAS1_0101_MR2__mpr-2", "oasis1:disc3:OAS1_0101_MR2:mpr-2"], ["oasis1__OAS1_0239_MR2__mpr-1", "oasis1:disc7:OAS1_0239_MR2:mpr-1"], ["oasis1__OAS1_0239_MR2__mpr-2", "oasis1:disc7:OAS1_0239_MR2:mpr-2"], ["ds005405__sub-016__3200", "ds005405/sub-016/ses-bhb/anat/sub-016_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-016__3201", "ds005405/sub-016/ses-glc/anat/sub-016_ses-glc_run-1_T1w.nii"], ["ds005405__sub-016__3202", "ds005405/sub-016/ses-glc/anat/sub-016_ses-glc_run-2_T1w.nii"], ["ds005405__sub-042__3203", "ds005405/sub-042/ses-bhb/anat/sub-042_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-042__3204", "ds005405/sub-042/ses-glc/anat/sub-042_ses-glc_run-1_T1w.nii"], ["ds005405__sub-042__3205", "ds005405/sub-042/ses-glc/anat/sub-042_ses-glc_run-2_T1w.nii"], ["ds005405__sub-098__3206", "ds005405/sub-098/ses-bhb/anat/sub-098_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-098__3207", "ds005405/sub-098/ses-glc/anat/sub-098_ses-glc_run-1_T1w.nii"], ["ds005405__sub-098__3208", "ds005405/sub-098/ses-glc/anat/sub-098_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0004_MR1__mpr-2", "oasis1:disc1:OAS1_0004_MR1:mpr-2"], ["oasis1__OAS1_0095_MR1__mpr-2", "oasis1:disc3:OAS1_0095_MR1:mpr-2"], ["oasis1__OAS1_0249_MR1__mpr-2", "oasis1:disc7:OAS1_0249_MR1:mpr-2"], ["oasis1__OAS1_0261_MR1__mpr-2", "oasis1:disc7:OAS1_0261_MR1:mpr-2"], ["oasis1__OAS1_0281_MR1__mpr-2", "oasis1:disc8:OAS1_0281_MR1:mpr-2"], ["oasis1__OAS1_0296_MR1__mpr-2", "oasis1:disc8:OAS1_0296_MR1:mpr-2"], ["oasis1__OAS1_0249_MR2__mpr-1", "oasis1:disc7:OAS1_0249_MR2:mpr-1"], ["oasis1__OAS1_0249_MR2__mpr-2", "oasis1:disc7:OAS1_0249_MR2:mpr-2"], ["ds005405__sub-027__3209", "ds005405/sub-027/ses-bhb/anat/sub-027_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-027__3210", "ds005405/sub-027/ses-glc/anat/sub-027_ses-glc_run-1_T1w.nii"], ["ds005405__sub-027__3211", "ds005405/sub-027/ses-glc/anat/sub-027_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0037_MR1__mpr-2", "oasis1:disc1:OAS1_0037_MR1:mpr-2"], ["oasis1__OAS1_0314_MR1__mpr-2", "oasis1:disc9:OAS1_0314_MR1:mpr-2"], ["oasis1__OAS1_0325_MR1__mpr-2", "oasis1:disc9:OAS1_0325_MR1:mpr-2"], ["ds005405__sub-023__3212", "ds005405/sub-023/ses-bhb/anat/sub-023_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-023__3213", "ds005405/sub-023/ses-glc/anat/sub-023_ses-glc_run-1_T1w.nii"], ["ds005405__sub-023__3214", "ds005405/sub-023/ses-glc/anat/sub-023_ses-glc_run-2_T1w.nii"], ["ds005405__sub-041__3215", "ds005405/sub-041/ses-bhb/anat/sub-041_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-041__3216", "ds005405/sub-041/ses-glc/anat/sub-041_ses-glc_run-1_T1w.nii"], ["ds005405__sub-041__3217", "ds005405/sub-041/ses-glc/anat/sub-041_ses-glc_run-2_T1w.nii"], ["ds005405__sub-044__3218", "ds005405/sub-044/ses-bhb/anat/sub-044_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-044__3219", "ds005405/sub-044/ses-glc/anat/sub-044_ses-glc_run-1_T1w.nii"], ["ds005405__sub-044__3220", "ds005405/sub-044/ses-glc/anat/sub-044_ses-glc_run-2_T1w.nii"], ["ds005405__sub-045__3221", "ds005405/sub-045/ses-bhb/anat/sub-045_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-045__3222", "ds005405/sub-045/ses-glc/anat/sub-045_ses-glc_run-1_T1w.nii"], ["ds005405__sub-045__3223", "ds005405/sub-045/ses-glc/anat/sub-045_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0121_MR1__mpr-2", "oasis1:disc4:OAS1_0121_MR1:mpr-2"], ["oasis1__OAS1_0218_MR1__mpr-2", "oasis1:disc6:OAS1_0218_MR1:mpr-2"], ["oasis1__OAS1_0227_MR1__mpr-2", "oasis1:disc6:OAS1_0227_MR1:mpr-2"], ["oasis1__OAS1_0333_MR1__mpr-2", "oasis1:disc9:OAS1_0333_MR1:mpr-2"], ["oasis1__OAS1_0386_MR1__mpr-2", "oasis1:disc11:OAS1_0386_MR1:mpr-2"], ["oasis1__OAS1_0387_MR1__mpr-2", "oasis1:disc11:OAS1_0387_MR1:mpr-2"], ["oasis1__OAS1_0395_MR1__mpr-2", "oasis1:disc11:OAS1_0395_MR1:mpr-2"], ["oasis1__OAS1_0395_MR2__mpr-1", "oasis1:disc11:OAS1_0395_MR2:mpr-1"], ["oasis1__OAS1_0395_MR2__mpr-2", "oasis1:disc11:OAS1_0395_MR2:mpr-2"], ["ds005405__sub-011__3224", "ds005405/sub-011/ses-bhb/anat/sub-011_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-011__3225", "ds005405/sub-011/ses-glc/anat/sub-011_ses-glc_run-1_T1w.nii"], ["ds005405__sub-011__3226", "ds005405/sub-011/ses-glc/anat/sub-011_ses-glc_run-2_T1w.nii"], ["ds005405__sub-028__3227", "ds005405/sub-028/ses-bhb/anat/sub-028_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-028__3228", "ds005405/sub-028/ses-glc/anat/sub-028_ses-glc_run-1_T1w.nii"], ["ds005405__sub-028__3229", "ds005405/sub-028/ses-glc/anat/sub-028_ses-glc_run-2_T1w.nii"], ["ds005405__sub-059__3230", "ds005405/sub-059/ses-bhb/anat/sub-059_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-059__3231", "ds005405/sub-059/ses-glc/anat/sub-059_ses-glc_run-1_T1w.nii"], ["ds005405__sub-059__3232", "ds005405/sub-059/ses-glc/anat/sub-059_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0079_MR1__mpr-2", "oasis1:disc2:OAS1_0079_MR1:mpr-2"], ["oasis1__OAS1_0080_MR1__mpr-2", "oasis1:disc2:OAS1_0080_MR1:mpr-2"], ["oasis1__OAS1_0108_MR1__mpr-2", "oasis1:disc3:OAS1_0108_MR1:mpr-2"], ["oasis1__OAS1_0117_MR1__mpr-2", "oasis1:disc4:OAS1_0117_MR1:mpr-2"], ["oasis1__OAS1_0147_MR1__mpr-2", "oasis1:disc4:OAS1_0147_MR1:mpr-2"], ["oasis1__OAS1_0151_MR1__mpr-2", "oasis1:disc5:OAS1_0151_MR1:mpr-2"], ["oasis1__OAS1_0377_MR1__mpr-2", "oasis1:disc10:OAS1_0377_MR1:mpr-2"], ["oasis1__OAS1_0396_MR1__mpr-2", "oasis1:disc11:OAS1_0396_MR1:mpr-2"], ["oasis1__OAS1_0406_MR1__mpr-2", "oasis1:disc11:OAS1_0406_MR1:mpr-2"], ["oasis1__OAS1_0413_MR1__mpr-2", "oasis1:disc11:OAS1_0413_MR1:mpr-2"], ["oasis1__OAS1_0080_MR2__mpr-1", "oasis1:disc2:OAS1_0080_MR2:mpr-1"], ["oasis1__OAS1_0080_MR2__mpr-2", "oasis1:disc2:OAS1_0080_MR2:mpr-2"], ["oasis1__OAS1_0117_MR2__mpr-1", "oasis1:disc4:OAS1_0117_MR2:mpr-1"], ["oasis1__OAS1_0117_MR2__mpr-2", "oasis1:disc4:OAS1_0117_MR2:mpr-2"], ["ds005405__sub-010__3233", "ds005405/sub-010/ses-bhb/anat/sub-010_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-010__3234", "ds005405/sub-010/ses-glc/anat/sub-010_ses-glc_run-1_T1w.nii"], ["ds005405__sub-010__3235", "ds005405/sub-010/ses-glc/anat/sub-010_ses-glc_run-2_T1w.nii"], ["ds005405__sub-019__3236", "ds005405/sub-019/ses-bhb/anat/sub-019_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-019__3237", "ds005405/sub-019/ses-glc/anat/sub-019_ses-glc_run-1_T1w.nii"], ["ds005405__sub-019__3238", "ds005405/sub-019/ses-glc/anat/sub-019_ses-glc_run-2_T1w.nii"], ["ds005405__sub-046__3239", "ds005405/sub-046/ses-bhb/anat/sub-046_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-046__3240", "ds005405/sub-046/ses-glc/anat/sub-046_ses-glc_run-1_T1w.nii"], ["ds005405__sub-046__3241", "ds005405/sub-046/ses-glc/anat/sub-046_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0006_MR1__mpr-2", "oasis1:disc1:OAS1_0006_MR1:mpr-2"], ["oasis1__OAS1_0025_MR1__mpr-2", "oasis1:disc1:OAS1_0025_MR1:mpr-2"], ["oasis1__OAS1_0051_MR1__mpr-2", "oasis1:disc2:OAS1_0051_MR1:mpr-2"], ["oasis1__OAS1_0104_MR1__mpr-2", "oasis1:disc3:OAS1_0104_MR1:mpr-2"], ["oasis1__OAS1_0131_MR1__mpr-2", "oasis1:disc4:OAS1_0131_MR1:mpr-2"], ["oasis1__OAS1_0136_MR1__mpr-2", "oasis1:disc4:OAS1_0136_MR1:mpr-2"], ["oasis1__OAS1_0141_MR1__mpr-2", "oasis1:disc4:OAS1_0141_MR1:mpr-2"], ["oasis1__OAS1_0264_MR1__mpr-2", "oasis1:disc7:OAS1_0264_MR1:mpr-2"], ["oasis1__OAS1_0392_MR1__mpr-2", "oasis1:disc11:OAS1_0392_MR1:mpr-2"], ["ds005405__sub-021__3242", "ds005405/sub-021/ses-bhb/anat/sub-021_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-021__3243", "ds005405/sub-021/ses-glc/anat/sub-021_ses-glc_run-1_T1w.nii"], ["ds005405__sub-021__3244", "ds005405/sub-021/ses-glc/anat/sub-021_ses-glc_run-2_T1w.nii"], ["ds005405__sub-029__3245", "ds005405/sub-029/ses-bhb/anat/sub-029_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-029__3246", "ds005405/sub-029/ses-glc/anat/sub-029_ses-glc_run-1_T1w.nii"], ["ds005405__sub-029__3247", "ds005405/sub-029/ses-glc/anat/sub-029_ses-glc_run-2_T1w.nii"], ["ds005405__sub-034__3248", "ds005405/sub-034/ses-bhb/anat/sub-034_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-034__3249", "ds005405/sub-034/ses-glc/anat/sub-034_ses-glc_run-1_T1w.nii"], ["ds005405__sub-034__3250", "ds005405/sub-034/ses-glc/anat/sub-034_ses-glc_run-2_T1w.nii"], ["ds005405__sub-049__3251", "ds005405/sub-049/ses-bhb/anat/sub-049_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-049__3252", "ds005405/sub-049/ses-glc/anat/sub-049_ses-glc_run-1_T1w.nii"], ["ds005405__sub-049__3253", "ds005405/sub-049/ses-glc/anat/sub-049_ses-glc_run-2_T1w.nii"], ["ds005405__sub-099__3254", "ds005405/sub-099/ses-bhb/anat/sub-099_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-099__3255", "ds005405/sub-099/ses-glc/anat/sub-099_ses-glc_run-1_T1w.nii"], ["ds005405__sub-099__3256", "ds005405/sub-099/ses-glc/anat/sub-099_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0038_MR1__mpr-2", "oasis1:disc1:OAS1_0038_MR1:mpr-2"], ["oasis1__OAS1_0097_MR1__mpr-2", "oasis1:disc3:OAS1_0097_MR1:mpr-2"], ["oasis1__OAS1_0111_MR1__mpr-2", "oasis1:disc3:OAS1_0111_MR1:mpr-2"], ["oasis1__OAS1_0140_MR1__mpr-2", "oasis1:disc4:OAS1_0140_MR1:mpr-2"], ["oasis1__OAS1_0148_MR1__mpr-2", "oasis1:disc4:OAS1_0148_MR1:mpr-2"], ["oasis1__OAS1_0152_MR1__mpr-2", "oasis1:disc5:OAS1_0152_MR1:mpr-2"], ["oasis1__OAS1_0153_MR1__mpr-2", "oasis1:disc5:OAS1_0153_MR1:mpr-2"], ["oasis1__OAS1_0174_MR1__mpr-2", "oasis1:disc5:OAS1_0174_MR1:mpr-2"], ["oasis1__OAS1_0193_MR1__mpr-2", "oasis1:disc6:OAS1_0193_MR1:mpr-2"], ["oasis1__OAS1_0202_MR1__mpr-2", "oasis1:disc6:OAS1_0202_MR1:mpr-2"], ["oasis1__OAS1_0346_MR1__mpr-2", "oasis1:disc9:OAS1_0346_MR1:mpr-2"], ["oasis1__OAS1_0370_MR1__mpr-2", "oasis1:disc10:OAS1_0370_MR1:mpr-2"], ["oasis1__OAS1_0410_MR1__mpr-2", "oasis1:disc11:OAS1_0410_MR1:mpr-2"], ["oasis1__OAS1_0416_MR1__mpr-2", "oasis1:disc11:OAS1_0416_MR1:mpr-2"], ["oasis1__OAS1_0435_MR1__mpr-2", "oasis1:disc12:OAS1_0435_MR1:mpr-2"], ["oasis1__OAS1_0442_MR1__mpr-2", "oasis1:disc12:OAS1_0442_MR1:mpr-2"], ["oasis1__OAS1_0111_MR2__mpr-1", "oasis1:disc3:OAS1_0111_MR2:mpr-1"], ["oasis1__OAS1_0111_MR2__mpr-2", "oasis1:disc3:OAS1_0111_MR2:mpr-2"], ["oasis1__OAS1_0202_MR2__mpr-1", "oasis1:disc6:OAS1_0202_MR2:mpr-1"], ["oasis1__OAS1_0202_MR2__mpr-2", "oasis1:disc6:OAS1_0202_MR2:mpr-2"], ["ds005405__sub-020__3257", "ds005405/sub-020/ses-bhb/anat/sub-020_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-020__3258", "ds005405/sub-020/ses-glc/anat/sub-020_ses-glc_run-1_T1w.nii"], ["ds005405__sub-020__3259", "ds005405/sub-020/ses-glc/anat/sub-020_ses-glc_run-2_T1w.nii"], ["ds005405__sub-052__3260", "ds005405/sub-052/ses-bhb/anat/sub-052_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-052__3261", "ds005405/sub-052/ses-glc/anat/sub-052_ses-glc_run-1_T1w.nii"], ["ds005405__sub-052__3262", "ds005405/sub-052/ses-glc/anat/sub-052_ses-glc_run-2_T1w.nii"], ["ds005405__sub-095__3263", "ds005405/sub-095/ses-bhb/anat/sub-095_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-095__3264", "ds005405/sub-095/ses-glc/anat/sub-095_ses-glc_run-1_T1w.nii"], ["ds005405__sub-095__3265", "ds005405/sub-095/ses-glc/anat/sub-095_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0092_MR1__mpr-2", "oasis1:disc3:OAS1_0092_MR1:mpr-2"], ["oasis1__OAS1_0125_MR1__mpr-2", "oasis1:disc4:OAS1_0125_MR1:mpr-2"], ["oasis1__OAS1_0132_MR1__mpr-2", "oasis1:disc4:OAS1_0132_MR1:mpr-2"], ["oasis1__OAS1_0144_MR1__mpr-2", "oasis1:disc4:OAS1_0144_MR1:mpr-2"], ["oasis1__OAS1_0189_MR1__mpr-2", "oasis1:disc5:OAS1_0189_MR1:mpr-2"], ["oasis1__OAS1_0209_MR1__mpr-2", "oasis1:disc6:OAS1_0209_MR1:mpr-2"], ["oasis1__OAS1_0224_MR1__mpr-2", "oasis1:disc6:OAS1_0224_MR1:mpr-2"], ["oasis1__OAS1_0232_MR1__mpr-2", "oasis1:disc7:OAS1_0232_MR1:mpr-2"], ["oasis1__OAS1_0246_MR1__mpr-2", "oasis1:disc7:OAS1_0246_MR1:mpr-2"], ["oasis1__OAS1_0277_MR1__mpr-2", "oasis1:disc8:OAS1_0277_MR1:mpr-2"], ["oasis1__OAS1_0302_MR1__mpr-2", "oasis1:disc8:OAS1_0302_MR1:mpr-2"], ["oasis1__OAS1_0311_MR1__mpr-2", "oasis1:disc9:OAS1_0311_MR1:mpr-2"], ["oasis1__OAS1_0348_MR1__mpr-2", "oasis1:disc9:OAS1_0348_MR1:mpr-2"], ["oasis1__OAS1_0353_MR1__mpr-2", "oasis1:disc10:OAS1_0353_MR1:mpr-2"], ["oasis1__OAS1_0368_MR1__mpr-2", "oasis1:disc10:OAS1_0368_MR1:mpr-2"], ["oasis1__OAS1_0385_MR1__mpr-2", "oasis1:disc11:OAS1_0385_MR1:mpr-2"], ["oasis1__OAS1_0394_MR1__mpr-2", "oasis1:disc11:OAS1_0394_MR1:mpr-2"], ["oasis1__OAS1_0408_MR1__mpr-2", "oasis1:disc11:OAS1_0408_MR1:mpr-2"], ["oasis1__OAS1_0420_MR1__mpr-2", "oasis1:disc12:OAS1_0420_MR1:mpr-2"], ["oasis1__OAS1_0421_MR1__mpr-2", "oasis1:disc12:OAS1_0421_MR1:mpr-2"], ["oasis1__OAS1_0431_MR1__mpr-2", "oasis1:disc12:OAS1_0431_MR1:mpr-2"], ["oasis1__OAS1_0437_MR1__mpr-2", "oasis1:disc12:OAS1_0437_MR1:mpr-2"], ["oasis1__OAS1_0448_MR1__mpr-2", "oasis1:disc12:OAS1_0448_MR1:mpr-2"], ["oasis1__OAS1_0092_MR2__mpr-1", "oasis1:disc3:OAS1_0092_MR2:mpr-1"], ["oasis1__OAS1_0092_MR2__mpr-2", "oasis1:disc3:OAS1_0092_MR2:mpr-2"], ["oasis1__OAS1_0353_MR2__mpr-1", "oasis1:disc10:OAS1_0353_MR2:mpr-1"], ["oasis1__OAS1_0353_MR2__mpr-2", "oasis1:disc10:OAS1_0353_MR2:mpr-2"], ["oasis1__OAS1_0368_MR2__mpr-1", "oasis1:disc10:OAS1_0368_MR2:mpr-1"], ["oasis1__OAS1_0368_MR2__mpr-2", "oasis1:disc10:OAS1_0368_MR2:mpr-2"], ["ds005405__sub-037__3266", "ds005405/sub-037/ses-bhb/anat/sub-037_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-037__3267", "ds005405/sub-037/ses-glc/anat/sub-037_ses-glc_run-1_T1w.nii"], ["ds005405__sub-037__3268", "ds005405/sub-037/ses-glc/anat/sub-037_ses-glc_run-2_T1w.nii"], ["ds005405__sub-048__3269", "ds005405/sub-048/ses-bhb/anat/sub-048_ses-bhb_run-2_T1w.nii"], ["ds005405__sub-048__3270", "ds005405/sub-048/ses-glc/anat/sub-048_ses-glc_run-1_T1w.nii"], ["ds005405__sub-048__3271", "ds005405/sub-048/ses-glc/anat/sub-048_ses-glc_run-2_T1w.nii"], ["oasis1__OAS1_0007_MR1__mpr-2", "oasis1:disc1:OAS1_0007_MR1:mpr-2"], ["oasis1__OAS1_0017_MR1__mpr-2", "oasis1:disc1:OAS1_0017_MR1:mpr-2"], ["oasis1__OAS1_0029_MR1__mpr-2", "oasis1:disc1:OAS1_0029_MR1:mpr-2"], ["oasis1__OAS1_0043_MR1__mpr-2", "oasis1:disc2:OAS1_0043_MR1:mpr-2"], ["oasis1__OAS1_0054_MR1__mpr-2", "oasis1:disc2:OAS1_0054_MR1:mpr-2"], ["oasis1__OAS1_0057_MR1__mpr-2", "oasis1:disc2:OAS1_0057_MR1:mpr-2"], ["oasis1__OAS1_0087_MR1__mpr-2", "oasis1:disc3:OAS1_0087_MR1:mpr-2"], ["oasis1__OAS1_0126_MR1__mpr-2", "oasis1:disc4:OAS1_0126_MR1:mpr-2"], ["oasis1__OAS1_0191_MR1__mpr-2", "oasis1:disc5:OAS1_0191_MR1:mpr-2"], ["oasis1__OAS1_0198_MR1__mpr-2", "oasis1:disc6:OAS1_0198_MR1:mpr-2"], ["oasis1__OAS1_0250_MR1__mpr-2", "oasis1:disc7:OAS1_0250_MR1:mpr-2"], ["oasis1__OAS1_0258_MR1__mpr-2", "oasis1:disc7:OAS1_0258_MR1:mpr-2"], ["oasis1__OAS1_0350_MR1__mpr-2", "oasis1:disc10:OAS1_0350_MR1:mpr-2"], ["oasis1__OAS1_0359_MR1__mpr-2", "oasis1:disc10:OAS1_0359_MR1:mpr-2"], ["oasis1__OAS1_0415_MR1__mpr-2", "oasis1:disc11:OAS1_0415_MR1:mpr-2"], ["oasis1__OAS1_0419_MR1__mpr-2", "oasis1:disc11:OAS1_0419_MR1:mpr-2"], ["oasis1__OAS1_0439_MR1__mpr-2", "oasis1:disc12:OAS1_0439_MR1:mpr-2"], ["oasis1__OAS1_0191_MR2__mpr-1", "oasis1:disc5:OAS1_0191_MR2:mpr-1"], ["oasis1__OAS1_0191_MR2__mpr-2", "oasis1:disc5:OAS1_0191_MR2:mpr-2"], ["oasis1__OAS1_0009_MR1__mpr-2", "oasis1:disc1:OAS1_0009_MR1:mpr-2"], ["oasis1__OAS1_0049_MR1__mpr-2", "oasis1:disc2:OAS1_0049_MR1:mpr-2"], ["oasis1__OAS1_0055_MR1__mpr-2", "oasis1:disc2:OAS1_0055_MR1:mpr-2"], ["oasis1__OAS1_0059_MR1__mpr-2", "oasis1:disc2:OAS1_0059_MR1:mpr-2"], ["oasis1__OAS1_0061_MR1__mpr-2", "oasis1:disc2:OAS1_0061_MR1:mpr-2"], ["oasis1__OAS1_0077_MR1__mpr-2", "oasis1:disc2:OAS1_0077_MR1:mpr-2"], ["oasis1__OAS1_0090_MR1__mpr-2", "oasis1:disc3:OAS1_0090_MR1:mpr-2"], ["oasis1__OAS1_0105_MR1__mpr-2", "oasis1:disc3:OAS1_0105_MR1:mpr-2"], ["oasis1__OAS1_0107_MR1__mpr-2", "oasis1:disc3:OAS1_0107_MR1:mpr-2"], ["oasis1__OAS1_0150_MR1__mpr-2", "oasis1:disc4:OAS1_0150_MR1:mpr-2"], ["oasis1__OAS1_0156_MR1__mpr-2", "oasis1:disc5:OAS1_0156_MR1:mpr-2"], ["oasis1__OAS1_0162_MR1__mpr-2", "oasis1:disc5:OAS1_0162_MR1:mpr-2"], ["oasis1__OAS1_0211_MR1__mpr-2", "oasis1:disc6:OAS1_0211_MR1:mpr-2"], ["oasis1__OAS1_0231_MR1__mpr-2", "oasis1:disc6:OAS1_0231_MR1:mpr-2"], ["oasis1__OAS1_0236_MR1__mpr-2", "oasis1:disc7:OAS1_0236_MR1:mpr-2"], ["oasis1__OAS1_0253_MR1__mpr-2", "oasis1:disc7:OAS1_0253_MR1:mpr-2"], ["oasis1__OAS1_0285_MR1__mpr-2", "oasis1:disc8:OAS1_0285_MR1:mpr-2"], ["oasis1__OAS1_0294_MR1__mpr-2", "oasis1:disc8:OAS1_0294_MR1:mpr-2"], ["oasis1__OAS1_0295_MR1__mpr-2", "oasis1:disc8:OAS1_0295_MR1:mpr-2"], ["oasis1__OAS1_0310_MR1__mpr-2", "oasis1:disc9:OAS1_0310_MR1:mpr-2"], ["oasis1__OAS1_0313_MR1__mpr-2", "oasis1:disc9:OAS1_0313_MR1:mpr-2"], ["oasis1__OAS1_0344_MR1__mpr-2", "oasis1:disc9:OAS1_0344_MR1:mpr-2"], ["oasis1__OAS1_0361_MR1__mpr-2", "oasis1:disc10:OAS1_0361_MR1:mpr-2"], ["oasis1__OAS1_0379_MR1__mpr-2", "oasis1:disc10:OAS1_0379_MR1:mpr-2"], ["oasis1__OAS1_0397_MR1__mpr-2", "oasis1:disc11:OAS1_0397_MR1:mpr-2"], ["oasis1__OAS1_0061_MR2__mpr-1", "oasis1:disc2:OAS1_0061_MR2:mpr-1"], ["oasis1__OAS1_0061_MR2__mpr-2", "oasis1:disc2:OAS1_0061_MR2:mpr-2"], ["oasis1__OAS1_0150_MR2__mpr-1", "oasis1:disc4:OAS1_0150_MR2:mpr-1"], ["oasis1__OAS1_0150_MR2__mpr-2", "oasis1:disc4:OAS1_0150_MR2:mpr-2"], ["oasis1__OAS1_0156_MR2__mpr-1", "oasis1:disc5:OAS1_0156_MR2:mpr-1"], ["oasis1__OAS1_0156_MR2__mpr-2", "oasis1:disc5:OAS1_0156_MR2:mpr-2"], ["oasis1__OAS1_0236_MR2__mpr-1", "oasis1:disc7:OAS1_0236_MR2:mpr-1"], ["oasis1__OAS1_0236_MR2__mpr-2", "oasis1:disc7:OAS1_0236_MR2:mpr-2"], ["oasis1__OAS1_0285_MR2__mpr-1", "oasis1:disc8:OAS1_0285_MR2:mpr-1"], ["oasis1__OAS1_0285_MR2__mpr-2", "oasis1:disc8:OAS1_0285_MR2:mpr-2"], ["oasis1__OAS1_0379_MR2__mpr-1", "oasis1:disc10:OAS1_0379_MR2:mpr-1"], ["oasis1__OAS1_0379_MR2__mpr-2", "oasis1:disc10:OAS1_0379_MR2:mpr-2"], ["oasis1__OAS1_0014_MR1__mpr-2", "oasis1:disc1:OAS1_0014_MR1:mpr-2"], ["oasis1__OAS1_0099_MR1__mpr-2", "oasis1:disc3:OAS1_0099_MR1:mpr-2"], ["oasis1__OAS1_0103_MR1__mpr-2", "oasis1:disc3:OAS1_0103_MR1:mpr-2"], ["oasis1__OAS1_0119_MR1__mpr-2", "oasis1:disc4:OAS1_0119_MR1:mpr-2"], ["oasis1__OAS1_0230_MR1__mpr-2", "oasis1:disc6:OAS1_0230_MR1:mpr-2"], ["oasis1__OAS1_0321_MR1__mpr-2", "oasis1:disc9:OAS1_0321_MR1:mpr-2"], ["oasis1__OAS1_0328_MR1__mpr-2", "oasis1:disc9:OAS1_0328_MR1:mpr-2"], ["oasis1__OAS1_0340_MR1__mpr-2", "oasis1:disc9:OAS1_0340_MR1:mpr-2"], ["oasis1__OAS1_0403_MR1__mpr-2", "oasis1:disc11:OAS1_0403_MR1:mpr-2"], ["oasis1__OAS1_0450_MR1__mpr-2", "oasis1:disc12:OAS1_0450_MR1:mpr-2"], ["oasis1__OAS1_0230_MR2__mpr-1", "oasis1:disc6:OAS1_0230_MR2:mpr-1"], ["oasis1__OAS1_0230_MR2__mpr-2", "oasis1:disc6:OAS1_0230_MR2:mpr-2"], ["oasis1__OAS1_0005_MR1__mpr-2", "oasis1:disc1:OAS1_0005_MR1:mpr-2"], ["oasis1__OAS1_0076_MR1__mpr-2", "oasis1:disc2:OAS1_0076_MR1:mpr-2"], ["oasis1__OAS1_0081_MR1__mpr-2", "oasis1:disc3:OAS1_0081_MR1:mpr-2"], ["oasis1__OAS1_0091_MR1__mpr-2", "oasis1:disc3:OAS1_0091_MR1:mpr-2"], ["oasis1__OAS1_0102_MR1__mpr-2", "oasis1:disc3:OAS1_0102_MR1:mpr-2"], ["oasis1__OAS1_0129_MR1__mpr-2", "oasis1:disc4:OAS1_0129_MR1:mpr-2"], ["oasis1__OAS1_0163_MR1__mpr-2", "oasis1:disc5:OAS1_0163_MR1:mpr-2"], ["oasis1__OAS1_0214_MR1__mpr-2", "oasis1:disc6:OAS1_0214_MR1:mpr-2"], ["oasis1__OAS1_0283_MR1__mpr-2", "oasis1:disc8:OAS1_0283_MR1:mpr-2"]]''')          # [[scan_id, s3_key], ...] for this part
CHUNK = 12


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


print("PART", PART, "scans", len(JOBS), flush=True)
r = sh("git clone --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer")
print("fastsurfer clone rc", r.returncode, flush=True)
r = sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas surfa")
print("pip rc", r.returncode, r.stderr[-200:], flush=True)
if sh("python -c 'import surfa'").returncode != 0:
    print("surfa from GitHub rc", sh("pip install -q git+https://github.com/freesurfer/surfa.git").returncode, flush=True)
os.makedirs("/tmp/fs/models", exist_ok=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip",
                           "/tmp/fs/mri_synthstrip")
urllib.request.urlretrieve("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
                           "/tmp/fs/models/synthstrip.1.pt")
src = open("/tmp/fs/mri_synthstrip").read().replace(
    "torch.load(modelfile, map_location=device)", "torch.load(modelfile, map_location=device, weights_only=False)")
open("/tmp/fs/mri_synthstrip", "w").write(src)
# batch runner: one python process per chunk calls SynthStrip's own main() for every image
# (no repeated interpreter start-up / torch import / CUDA initialisation)
open("/tmp/fs/ss_batch.py", "w").write(
    "import sys, runpy" + chr(10) +
    "ns = runpy.run_path('/tmp/fs/mri_synthstrip', run_name='synthstrip_lib')" + chr(10) +
    "extra = sys.argv[2:]" + chr(10) +
    "for line in open(sys.argv[1]).read().splitlines():" + chr(10) +
    "    if not line.strip(): continue" + chr(10) +
    "    img, msk = line.split(chr(9))" + chr(10) +
    "    sys.argv = ['mri_synthstrip', '-i', img, '-m', msk] + extra" + chr(10) +
    "    try: ns['main']()" + chr(10) +
    "    except SystemExit: pass" + chr(10) +
    "    except Exception as e: print('SYNTHSTRIP FAILED', img, e, flush=True)" + chr(10))
os.environ["FREESURFER_HOME"] = "/tmp/fs"

import nibabel as nib
import numpy as np
try:
    import torch
    NGPU = max(1, torch.cuda.device_count()) if "cuda" in "--device cuda" else 1
except Exception:
    NGPU = 1
print("GPUs used:", NGPU, flush=True)

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["scan"] for l in open(OUT)}
todo = [j for j in JOBS if j[0] not in done]
chunks = [todo[k:k + CHUNK] for k in range(0, len(todo), CHUNK)]


O1_LOCK = {}
O1_GUARD = threading.Lock()


def oasis1_file(key):
    """key = oasis1:disc<N>:<SESSION>:mpr-<k>. Streams the WashU disc tarball once per kernel and
    keeps only the RAW MPRAGE files of the sessions this kernel needs; returns the .hdr path."""
    _, disc, sess, mpr = key.split(":")
    out = f"/tmp/o1/{sess}_{mpr}_anon.hdr"
    with O1_GUARD:
        lock = O1_LOCK.setdefault(disc, threading.Lock())
    with lock:
        if not os.path.exists(f"/tmp/o1/.{disc}.done"):
            import tarfile
            need = {(k.split(":")[2], k.split(":")[3]) for _, k in JOBS if k.startswith(f"oasis1:{disc}:")}
            os.makedirs("/tmp/o1", exist_ok=True)
            url = f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_{disc}.tar.gz"
            with urllib.request.urlopen(url, timeout=600) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
                for m in tf:
                    b = os.path.basename(m.name)
                    parts = b.split("_")
                    if m.isfile() and "/RAW/" in m.name and len(parts) >= 4:
                        if (f"{parts[0]}_{parts[1]}_{parts[2]}", parts[3]) in need:
                            with open(f"/tmp/o1/{b}", "wb") as fo:
                                fo.write(tf.extractfile(m).read())
            open(f"/tmp/o1/.{disc}.done", "w").write("ok")
    return out


def fetch_one(job, d):
    sid, key = job
    if key.startswith("oasis1:"):
        try:
            h = oasis1_file(key)
            im = nib.load(h)
            arr = np.asanyarray(im.dataobj).astype(np.float32).squeeze()
            # OASIS-1 RAW is Analyze with orient=2 (sagittal), which nibabel ignores (it assumes axial):
            # axis0 posterior->anterior, axis1 inferior->superior, axis2 sagittal slices (1.25 mm, R->L: checked on the cohort's L/R asymmetries)
            aff = np.array([[0, 0, -1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
            aff[:3, 3] = -aff[:3, :3] @ (np.array(arr.shape) / 2.0)
            nib.save(nib.Nifti1Image(arr, aff), f"{d}/{sid}.nii.gz")
            json.dump({"Manufacturer": "Siemens", "ManufacturersModelName": "Vision", "MagneticFieldStrength": 1.5,
                       "o1_orient": "ASL-v2"}, open(f"{d}/{sid}.meta.json", "w"))
            return True
        except Exception as e:
            print(f"  oasis1 fail {sid}: {type(e).__name__} {e}", flush=True)
            return False
    for attempt in range(4):
        try:
            p = f"{d}/{sid}.nii.gz"
            url = key if key.startswith("http") else S3 + key       # OpenBHB lives on HuggingFace
            if key.endswith(".nii.gz"):
                urllib.request.urlretrieve(url, p)
            else:
                urllib.request.urlretrieve(url, f"{d}/{sid}.nii")
                with open(f"{d}/{sid}.nii", "rb") as fi, gzip.open(p, "wb", compresslevel=1) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(f"{d}/{sid}.nii")
            meta = {}
            ds = key.split("/")[0]
            for side in (() if key.startswith("http") else (key.rsplit(".nii", 1)[0] + ".json", f"{ds}/T1w.json")):
                try:
                    meta = json.loads(urllib.request.urlopen(S3 + side, timeout=60).read())
                    break
                except Exception:
                    pass
            keep = {k: meta.get(k) for k in ("Manufacturer", "ManufacturersModelName", "MagneticFieldStrength",
                                             "SoftwareVersions", "InstitutionName")}
            json.dump(keep, open(f"{d}/{sid}.meta.json", "w"))
            return True
        except Exception as e:
            print(f"  download retry {sid}: {type(e).__name__}", flush=True)
            time.sleep(10)
    return False


def fetch_chunk(ch, d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch_one(j, d), ch))


n_ok = len(done)
dirs = ["/tmp/dlA", "/tmp/dlB"]
pre = None
if chunks:
    fetch_chunk(chunks[0], dirs[0])
last_dt = 0.0
for c, ch in enumerate(chunks):
    if time.time() - T0 + 1.15 * last_dt > BUDGET_S:
        print("time budget reached, stopping", flush=True)
        break
    d = dirs[c % 2]
    if pre is not None:
        pre.join()
    if c + 1 < len(chunks):
        pre = threading.Thread(target=fetch_chunk, args=(chunks[c + 1], dirs[(c + 1) % 2]))
        pre.start()
    t1 = time.time()
    shutil.rmtree("/tmp/out", ignore_errors=True)
    # one FastSurfer process per GPU (Kaggle "NvidiaTeslaT4" = 2 x T4), scans split between them
    imgs = sorted(f for f in os.listdir(d) if f.endswith(".nii.gz"))
    groups = [imgs[g::NGPU] for g in range(NGPU)]
    procs = []
    for g, grp in enumerate(groups):
        if not grp:
            continue
        gd = f"{d}/gpu{g}"
        os.makedirs(gd, exist_ok=True)
        for f in grp:
            if not os.path.exists(f"{gd}/{f}"):
                os.symlink(f"{d}/{f}", f"{gd}/{f}")
        cmd = ("cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python FastSurferCNN/run_prediction.py "
               f"--in_dir {gd} --tag '*.nii.gz' --remove_suffix .nii.gz --sd /tmp/out "
               f"--device cuda --viewagg_device cuda --threads {max(1, 4 // NGPU)} --vox_size 1 --batch_size 8")
        procs.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}))
    outs = [p.communicate() for p in procs]
    r = subprocess.CompletedProcess("fastsurfer", 0, "".join(o[0] for o in outs), "".join(o[1] for o in outs))
    with open(f"{d}/ss_list.txt", "w") as fh:
        for sid, key in ch:
            if os.path.exists(f"{d}/{sid}.nii.gz"):
                fh.write(f"{d}/{sid}.nii.gz" + chr(9) + f"{d}/{sid}.ssmask.nii.gz" + chr(10))
    rs = sh(f"python /tmp/fs/ss_batch.py {d}/ss_list.txt -g")
    if not any(os.path.exists(f"{d}/{sid}.ssmask.nii.gz") for sid, _ in ch):   # fallback: per scan
        for sid, _ in ch:
            if os.path.exists(f"{d}/{sid}.nii.gz"):
                rs = sh(f"python /tmp/fs/mri_synthstrip -i {d}/{sid}.nii.gz -m {d}/{sid}.ssmask.nii.gz -g")
    rows = []
    for sid, key in ch:
        segp = f"/tmp/out/{sid}/mri/aparc.DKTatlas+aseg.deep.mgz"
        if not os.path.exists(segp):
            print(f"  FAIL seg {sid}", flush=True)
            continue
        try:
            s = nib.load(segp)
            a = np.asanyarray(s.dataobj).astype(np.int32)
            vox = float(np.prod(s.header.get_zooms()[:3]))
            labs, cnt = np.unique(a[a > 0], return_counts=True)
            im = nib.load(f"{d}/{sid}.nii.gz")
            row = {"scan": sid, "key": key, "zooms_in": [float(z) for z in im.header.get_zooms()[:3]],
                   "shape_in": [int(x) for x in im.shape[:3]]}
            row.update(json.load(open(f"{d}/{sid}.meta.json")) if os.path.exists(f"{d}/{sid}.meta.json") else {})
            mp = f"/tmp/out/{sid}/mri/mask.mgz"
            if os.path.exists(mp):
                row["mask_vol"] = float((np.asanyarray(nib.load(mp).dataobj) > 0).sum()) * vox
            mpath = f"{d}/{sid}.ssmask.nii.gz"
            if os.path.exists(mpath):
                mm = nib.load(mpath)
                row["icv_synthstrip"] = float((np.asanyarray(mm.dataobj) > 0).sum()) * float(np.prod(mm.header.get_zooms()[:3]))
            else:
                print(f"  synthstrip missing {sid}: {rs.stdout[-200:]} {rs.stderr[-200:]}", flush=True)
            row.update({f"V{int(l)}": float(n) * vox for l, n in zip(labs, cnt)})
            rows.append(row)
        except Exception as e:
            print(f"  ERROR {sid}: {type(e).__name__} {e}", flush=True)
    with open(OUT, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + chr(10))
    n_ok += len(rows)
    dt = time.time() - t1
    last_dt = dt
    print(f"chunk {c+1}/{len(chunks)}: {len(rows)}/{len(ch)} ok, {dt:.0f}s ({dt/max(len(ch),1):.1f}s/scan), "
          f"total {n_ok}/{len(JOBS)}, elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
    if len(rows) == 0:
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
        if c == 0:
            break
print(f"finished: {n_ok} rows, {(time.time()-T0)/3600:.2f} h", flush=True)
