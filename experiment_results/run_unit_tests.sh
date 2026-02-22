#!/bash/bin

model-train --cfg az_config_unit_test.json
model-eval --trained_model_dir bc-azel-test/ --evaluation_name sample_eval --specific_years 2016 2017 --specific_months 12 --specific_days 7 21 

model-train --cfg az_grid_config_unit_test.json
model-eval --trained_model_dir bc-azel-grid-test/ --evaluation_name sample_eval --specific_years 2016 2017 --specific_months 12 --specific_days 7 21 

model-train --cfg ra_config_unit_test.json
model-eval --trained_model_dir bc-radec-test/ --evaluation_name sample_eval --specific_years 2016 2017 --specific_months 12 --specific_days 7 21 

model-train --cfg ra_grid_config_unit_test.json
model-eval --trained_model_dir bc-radec-grid-test/ --evaluation_name sample_eval --specific_years 2016 2017 --specific_months 12 --specific_days 7 21
