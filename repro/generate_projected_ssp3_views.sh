# Define initial input image
input_path=data/predictions/SSP3/2099SSP3-to-2100SSP3.npy

# Iterate over yearly transition configuration files
for config_path in config/projections/SSP3/*
do
  filename="$(basename -s .yaml $config_path).npy"
  output_path=data/predictions/projection_SSP3/$filename
  python run_image_distortion.py --cfg=$config_path --input=$input_path --o=$output_path
  input_path=$output_path
done
