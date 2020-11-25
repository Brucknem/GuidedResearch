### Setup ### 

Run the setup.sh script:

```bash setup.sh```

This installs a python virtualenv that is used to run the script

### Source virtualenv ###

```source venv/bin/activate```

### Run script ### 

```python android_sensors_visualization.py <path_to_folder_with_csvs> <output_filename> [opt_number_of_plots_per_line]```

### Small test ###

```python android_sensors_visualization.py test_data jitter_visualization 3```
