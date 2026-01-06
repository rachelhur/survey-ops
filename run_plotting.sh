# original command -- test
# python /home/rachel/Projects/survey-ops/survey_ops/plotting.py -o my_test_schedule.gif -f /home/rachel/Projects/survey-ops/data/fields2radec.json -s /home/rachel/Projects/survey-ops/data/true_schedule.csv


# python /home/rachel/Projects/survey-ops/survey_ops/plotting.py -o /home/rachel/Projects/survey-ops/results/v2f-environment-testing/original_field_schedule.gif -f /home/rachel/Projects/survey-ops/data/fields2radec.json -s /home/rachel/Projects/survey-ops/data/true_schedule.csv
# Plot field schedule
    # original schedule:
# python /home/rachel/Projects/survey-ops/survey_ops/plotting.py -o /home/rachel/Projects/survey-ops/results/v2f-environment-testing/original_field_schedule.gif -f /home/rachel/Projects/survey-ops/data/field2radec.json -s /home/rachel/Projects/survey-ops/results/v2f-environment-testing/original_schedule.csv
    # policy roll-out schedule:
# python /home/rachel/Projects/survey-ops/survey_ops/plotting.py -o /home/rachel/Projects/survey-ops/results/v2f-environment-testing/eval_field_schedule.gif -f /home/rachel/Projects/survey-ops/data/field2radec.json -s /home/rachel/Projects/survey-ops/results/v2f-environment-testing/eval_schedule.csv
# plot bins:
python /home/rachel/Projects/survey-ops/survey_ops/plotting.py -f /home/rachel/Projects/survey-ops/data/nside16_bin2radec.json -o /home/rachel/Projects/survey-ops/results/v2f-environment-testing/bin_schedule.gif -b /home/rachel/Projects/survey-ops/results/v2f-environment-testing/bin_schedule.csv -n 16
