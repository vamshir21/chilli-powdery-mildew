# Chilli Powdery Mildew Detection & Severity Assessment

## Problem
Automated detection and multi-stage severity assessment of chilli powdery mildew under real-field lighting conditions.

## Dataset
- Field-captured chilli leaf images
- Disease severity based on standard 0–9 disease scale
- Mapped to 4 classes: Healthy, Mild, Moderate, Severe
- Datasets are not versioned in Git

## Pipeline
1. Dataset curation & stratified split (70/15/15)
2. Lighting normalization using CLAHE (LAB color space)
3. Leaf segmentation (U-Net) — upcoming
4. Severity estimation based on infected area — upcoming

## Repository Rules
- No datasets in Git
- One feature = one branch
- Pull before push

## Status
Dataset preparation and preprocessing completed.
