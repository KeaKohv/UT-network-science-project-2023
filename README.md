# UT-network-science-project-2023

Project for UT Network Science course spring 2023.

## Installation
### Prerequisites
If you wish to run the code on a GPU, you should have CUDA 11.7 installed.
To use other CUDA versions, consult the documentation for PyTorch and PyTorch Geometric

1. Installing the Dependencies
   ```
    pip3 install torch torchvision torchaudio
    pip3 install torch_geometric
    pip3 install pandas
    pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
   ```
2. Unpack `data/subsamples.zip` into `data/`
3. Learn How to use ngcf_final.py
   ```
   $ python ngcf_final.py --help
   usage: ngcf_final.py [-h] --file FILE --split {random,one-out,timeline}
   
   options:
     -h, --help            show this help message and exit
     --file FILE
     --split {random,one-out,timeline}
   ```
4. Run ngcf_final.py using split strategy random on the first subsample
   ```
   python ngcf_final.py --split random --file data/1_subsample_2018_2019_2000_users.csv
   ```
5. To run all of the experiments we performed:
   1. Create the directory `results_final`
   2. Run all of the commands in `experiments.sh`:
      ```
      time python ngcf_final.py --split random --file data/1_subsample_2018_2019_2000_users.csv > results_final/random_1.txt
      time python ngcf_final.py --split random --file data/2_subsample_2018_2019_2000_users_pruned.csv > results_final/random_2.txt
      time python ngcf_final.py --split random --file data/3_subsample_2018_2019_1000_users.csv > results_final/random_3.txt
      time python ngcf_final.py --split random --file data/4_subsample_2018_2019_1000_users_pruned.csv > results_final/random_4.txt
      time python ngcf_final.py --split one-out --file data/1_subsample_2018_2019_2000_users.csv > results_final/one-out_1.txt
      time python ngcf_final.py --split one-out --file data/2_subsample_2018_2019_2000_users_pruned.csv > results_final/one-out_2.txt
      time python ngcf_final.py --split one-out --file data/3_subsample_2018_2019_1000_users.csv > results_final/one-out_3.txt
      time python ngcf_final.py --split one-out --file data/4_subsample_2018_2019_1000_users_pruned.csv > results_final/one-out_4.txt
      time python ngcf_final.py --split timeline --file data/1_subsample_2018_2019_2000_users.csv > results_final/timeline_1.txt
      time python ngcf_final.py --split timeline --file data/2_subsample_2018_2019_2000_users_pruned.csv > results_final/timeline_2.txt
      time python ngcf_final.py --split timeline --file data/3_subsample_2018_2019_1000_users.csv > results_final/timeline_3.txt
      time python ngcf_final.py --split timeline --file data/4_subsample_2018_2019_1000_users_pruned.csv > results_final/timeline_4.txt
      ```
