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
