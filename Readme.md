```
%
% SPRLT: Local Transformer With Spatial Partition Restore for Hyperspectral Image Classification.
%
%    This demo shows the SPRLT model for hyperspectral image classification.
%
%    main.py ....... A main script executing experiments upon IP, PU, and HU data sets.
%    data.py ....... A script implementing various data manipulation functions.
%    util.py ....... A script implementing the sample selection function and etc.
%    sprltNet.py ....... A script implementing the sprlt models.
%    train_test.py ....... A script implementing the training function and test function etc.
%	 visualization.py ....... A script implementing the visualization for model.

%    /datasets ............... The folder including the IP, PU, and HU data sets, where "DS" represents spatially disjoint.
%	 /models ...............The folder storing training models.
%    /classification_maps .......... The folder storing classification results.
%    
%   --------------------------------------
%   Note: Required core python libraries
%   --------------------------------------
%   1. python 3.6.5
%   2. pytorch >= 1.7.0
%   3. einops
%   4. scipy = 1.5.4
%   5. opencv-python 4.4.0.46
%   6. sklearn
%   7. matplotlib = 3.3.0
%   8. numpy = 1.19.2
%   --------------------------------------
%   Cite:
%   --------------------------------------
%
%   [1] Z. Xue, Q. Xu and M. Zhang, "Local Transformer with Spatial Partition Restore for Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2022.3174135.
%
%   --------------------------------------
%   Copyright & Disclaimer
%   --------------------------------------
%
%   The programs contained in this package are granted free of charge for
%   research and education purposes only. 
```