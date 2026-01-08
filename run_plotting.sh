#!/bin/bash
current_dir=$(pwd)
result_dir="v2f-environment-testing"

# original command -- test
# python /home/rachel/Projects/survey-ops/survey_ops/plotting.py -o my_test_schedule.gif -f /home/rachel/Projects/survey-ops/data/fields2radec.json -s /home/rachel/Projects/survey-ops/data/true_schedule.csv

# Plot field schedule
    # original schedule:
# echo "Creating original schedule gif"
# python $current_dir/survey_ops/plotting.py -o $current_dir/results/v2f-environment-testing/original_field_schedule.gif -f $current_dir/data/field2radec.json -s $current_dir/results/v2f-environment-testing/original_schedule.csv
#     # policy roll-out schedule:
# echo "Creating policy roll-out schedule gif"
# python $current_dir/plotting.py -o $current_dir/results/v2f-environment-testing/eval_field_schedule.gif -f $current_dir/data/field2radec.json -s $current_dir/results/v2f-environment-testing/eval_schedule.csv
# plot bins:
echo "Creating bin schedule gif"
python "$current_dir/survey_ops/plotting.py" -f "$current_dir/data/nside32_bin2radec.json" -o "$current_dir/results/$result_dir/bin_schedule.gif" -b "$current_dir/results/$result_dir/bin_schedule.csv" -n 32