# SVM model
Repository to transfer the SVM model to openEO to automatically create training for snow and ice classification.

Auxiliary folder contains scripts to create the required data for create_trainings function (water mask, cloud mask, NDVI, NDSI, solar_incidence_angle, shadow index, distance index, and blue and nir difference).


snow_SVM/
│
├── auxiliary/
│   ├── altitude_mask_udf.py                      
│   ├── aspect_udf.py
│   ├── distance_udf.py
│   ├── scale_distance_udf.py
│   ├── slope_udf.py
│   ├── cloud_water_mask.ipynb
│   ├── distance_index.ipynb
│   ├── indices.ipynb
│   ├── solar_angle.ipynb
