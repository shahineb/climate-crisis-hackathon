# Define initial input image
input_path=data/predictions/SSP1/2099SSP1-to-2100SSP1.npy

# Iterate over yearly transition configuration files
for config_path in config/projections/SSP1/*
do
  filename="$(basename -s .yaml $config_path).npy"
  output_path=data/predictions/projection_SSP1/$filename
  python run_image_distortion.py --cfg=$config_path --input=$input_path --o=$output_path
  input_path=$output_path
done
