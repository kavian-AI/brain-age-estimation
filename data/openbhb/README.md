---
license: apache-2.0
pretty_name: op
size_categories:
- 1K<n<10K
---

**OpenBHB: a Multi-Site Brain MRI Dataset for Age Prediction and Debiasing**

The Open Big Healthy Brains (OpenBHB) dataset is a large (N>5000) multi-site 3D brain MRI dataset gathering 10 public datasets (IXI, ABIDE 1, ABIDE 2, CoRR, GSP, Localizer, MPI-Leipzig, NAR, NPC, RBP) of T1 images acquired across 93 different centers, spread worldwide (North America, Europe and China). Only healthy controls have been included in OpenBHB with age ranging from 6 to 88 years old, balanced between males and females. All T1 images have been uniformly pre-processed with CAT12 (SPM), FreeSurfer (FSL) and Quasi-Raw (in-house minimal pre-processing) and they all passed a visual quality check. Both Voxel-Based Morphometry and Surface-Based Morphometry measures are available for each T1 MRI. Participant's age and sex are provided as well as the acquisition site, MRI magnetic field and MRI scanner settings used for each image acquisition. 

*Note:* OpenBHB has been divided into an official train, validation and test split for the open challenge currently deployed on brain age prediction and site-effect removal (see below). To avoid any data leakage during this challenge, data in test are kept private on the submission servers to compute the challenge metrics. Only training and validation data are openly available for now.

**Age prediction with site-effect removal challenge** 

OpenBHB has been designed for **brain age prediction** and debiasing with **site-effect removal** in current brain MRI datasets through representation learning. The challenge consists in developing new algorithms taking as input T1 MRI images available in OpenBHB and outputting representation vectors preserving the biological variability (age) and removing undesirable non-biological confounding variables (acquisition site/settings). The representation quality is evaluated through linear probing on brain age prediction and site debiasing with (MAE and RMSE for age, balanced accuracy for site prediction). All algorithms can be submitted on [RAMP](https://ramp.studio/problems/brain_age_with_site_removal) with a public recording of their performance and an official leaderboard.

**Dataset organization** 

This dataset comprises 3227 training samples, 757 validation samples and 664 testing samples (kept private) dedicated to the OpenBHB challenge. Each sample comes with original T1 MRI and **6 pre-processed numpy arrays**: 
- quasi-raw volume, linearly co-registered, skull-stripped, bias-corrected
- cat12 volume, non-linearly co-registered, gray-matter extracted
- surface-based features computed on mesh (xhemi) using FreeSurfer 
- cat12 Region-Of-Interests, gray matter volumes computed per brain region using the Neuromorphometric atlas
- FreeSurfer Desikan ROI, surface-based features computed using Desikan atlas
- FreeSurfer Destrieux ROI (surface-based features computed using Destrieux atlas)

``` 
openBHB/ 
├── README.md
├── participants.tsv # meta-data relative to subjects and scans, e.g. age, sex, site label
├── qc.tsv # quality check metrics, e.g. Euler number
├── train/
    ├──derivatives/
        ├──sub-*
            ├──ses-1
                ├── sub-*_preproc-quasiraw_T1w.npy          # quasi-raw 3D volume
                ├── sub-*_preproc-cat12vbm_desc-gm_T1w.npy  # cat12 3D volume
                ├── sub-*_preproc-cat12vbm_desc-gm_ROI.npy       # gray matter volume on Neuromorphometric atlas
                ├── sub-*_preproc-freesurfer_desc-xhemi_T1w.npy  # surface-based features on mesh
                ├── sub-*_preproc-freesurfer_desc-desikan_ROI.npy # surface-based features on Desikan atlas
                ├── sub-*_preproc-freesurfer_desc-destrieux_ROI.npy # surface-based features on Destrieux atlas
        ├──cat12vbm_roi/
            cat12vbm_roi_features.csv   # GM volume on Neuromorphometric atlas concatenated across subjects
        ├──freesurfer_roi/
            desikan_roi_features.csv    # surface-based features on Desikan atlas concatenated across subjects
            destrieux_roi_features.csv  # surface-based features on Destrieux atlas concatenated across subjects
    ├──rawdata/
        ├──sub-*
            ├──ses-1
                ├── sub-*_T1w.nii # raw T1 MRI
├── val/
    ├──derivatives/
        ├──sub-*
            ...  # same structure as 'train'
    ├──rawdata/
        ├──sub-*
            ... # same structure as 'train'
├── resource/
    ├── cat12vbm_space-MNI152_desc-gm_TPM.nii.gz  # template used in CAT12 co-registration
    ├── quasiraw_space-MNI152_desc-brain_T1w.nii.gz # template used in quasi-raw co-registration
    ├── cat12vbm_labels.txt  # ROI names for Neuromorphometric atlas
    ├── freesurfer_atlas-desikan_labels.txt # ROI names for Desikan atlas
    ├── freesurfer_atlas-destrieux_labels.txt # ROI names for Destrieux atlas
    ├── freesurfer_channels.txt # Feature names derived by FreeSurfer on Destrieux/Desikan atlas
    ├── freesurfer_xhemi_channels.txt 
    ├── neuromorphometrics.csv  # Neuromorphometric labels
    ├── neuromorphometrics.nii  # Neuromorphometric atlas
    ├── resources.json  # meta-data info on numpy arrays
``` 

**Acknowledgments**

If you use this dataset for your work, please use the following citation:

```bibtex
@article{dufumier2021,
  title={{OpenBHB: a Large-Scale Multi-Site Brain MRI Data-set for Age Prediction and Debiasing}},
  author={Dufumier, Benoit and Grigis, Antoine and Victor, Julie and Ambroise, Corentin and Frouin, Vincent and Duchesnay, Edouard},
  journal={NeuroImage},
  year={2022}
}
```

**Licence and Data Usage Agreement**

This dataset is under Licence CC BY-NC-SA 3.0. By downloading this dataset, you also agree to the most restrictive Data Usage Agreement (DUA) of all cohorts:

| Dataset                                                                                                           | License Terms        |
| ----------------------------------------------------------------------------------------------------------------- | -------------------- |
| [**ABIDE 1**](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)                                        | CC BY-NC-SA 3.0 (SA) |
| [**ABIDE 2**](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html)                                       | CC BY-NC-SA 3.0      |
| [**IXI**](https://brain-development.org/ixi-dataset)                                                              | CC0                  |
| [**CoRR**](https://www.nature.com/articles/sdata201449)                                                      | CC0                  |
| [**GSP**](https://www.nature.com/articles/sdata201531)                                                                      | CC0                    |
| [**NAR**](https://doi.org/10.18112/openneuro.ds002345.v1.0.1)                                                     | CC0                  |
| [**MPI-Leipzig**](https://doi.org/10.1038/sdata.2018.308)                                                         | CC0                  |
| [**NPC**](https://doi.org/10.18112/openneuro.ds002330.v1.1.0)                                                     | CC0                  |
| [**RBP**](https://doi.org/10.1016/j.jneuroling.2018.03.005)    | CC0                  |
| [**Localizer**](https://www.sciencedirect.com/science/article/pii/S1053811915008745)                                                 | CC BY 3.0            |
