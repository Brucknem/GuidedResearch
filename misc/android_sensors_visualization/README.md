### Setup ### 

1. Install the Android App [Sensor Record](https://play.google.com/store/apps/details?id=de.martingolpashin.sensor_record&hl=gsw&gl=US)
2. Open it and record some data.
3. Send the data to your PC running an \*nix based OS 
4. Extract the zipped data folder including the recorded .csv files

***

5. Run the setup.sh script:

```bash setup.sh```

This installs a python virtualenv that is used to run the script

### Source virtualenv ###

```source venv/bin/activate```

### Run script ### 

```python android_sensors_visualization.py <path_to_folder_with_csvs> <output_filename> [opt_number_of_plots_per_line]```

### Small test ###

```python android_sensors_visualization.py test_data jitter_visualization 3```
