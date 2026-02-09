# 3DCNN ArtifactReduction



## Getting started

Repository for the 3DCNN network. 
Basic idea: extract feature maps from 2D slice, stack them and reconstruct volume with 3D decoder.

Usage: 
- 1_train2D.py - Train the 2D U-Net
- 2_extract_features.py - Use the trained 2D U-Net for feature extraction
- 3_train3D.py - Train a 3D decoder on the extracted features.
