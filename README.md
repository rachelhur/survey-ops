Code for a reinforcement learning based agent capable of optimizing telescope scheduling at BLANCO.

A simple behavior cloning agent can be trained by running

python train.py --fits_path ../data/decam-exposures-20251211.fits --algorithm_name behavior_cloning --specific_years 2018 --specific_months 1 --do_cyclical_norm --do_max_norm --do_inverse_airmass --remove_large_time_diffs

*note: the above command will only run w/o error if the fits file exists in <fits_path>. results will be saved in '../experiment_results/test_experiment' unless otherwise specified. See train.py --help for detailed argument descriptions

The trained agent can roll out their policy for any of the nights available in the fits file, and several movies of the expert and and agent's schedule will be output. 

python eval.py --trained_model_dir ../experiment_results/test_experiment/ --specific_years 2017 --specific_months 1 --specific_days 1 2 3 4 5 --evaluation_name test_run

The results and movies will be saved in `<trained_model_dir>/<evaluation_name>/<date>/`

If there is no data for the given dates, then the script will produce an error.
