start with 
. ./vc_demo.sh try_1 path_to_cmu/cmu_arctic


clean feature every time you run the code, We have to modify the code to overwrite them every time 

after the code is completed:
features dir will have the all alingned features in npy array
generated/try_1 will have all generated audio
models checkpoint will be in checkpoints 

open evaluation_notebook.ipynb, it will make evaluation more clear and interactive

Change model hyperparameter at hparam.py and it will reflect in all other necessary files automatically. 

link to paper
https://ieeexplore.ieee.org/document/8063435/
