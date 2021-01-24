"""
Description : Run inference with CNN to predict projected temperature fields in a recurring fashion

Usage: run_cnn_inference.py --root=<path_to_dataset_directory> --weights=<path_to_model_weights> --n=<nb_of_time_steps> --o=<path_to_dumping_directory>

Options:
  --root=<path_to_dataset_directory>          Path to dataset directory
  --weights=<path_to_model_weights>           Path to model weights to use for prediction
  --n=<nb_of_time_steps>                      Number of time steps to take in the future
  --o=<path_to_model_weigths>                 Path to dumping directory
"""
import os
import logging
from docopt import docopt
import numpy as np
import xarray as xr
import torch
from progress.bar import Bar
from run_cnn_training import ConvNet, FieldsDataset


def main(args):
    dataset = FieldsDataset(root=args['--root'])
    model = load_model(weights_path=args['--weights'])

    predict_next_frames(model=model,
                        dataset=dataset,
                        n_steps=int(args['--n']),
                        dump_dir=args['--o'])


def load_model(weights_path):
    model = ConvNet(in_channels=3, out_channels=1, hidden_channels=8)
    model.load_state_dict(torch.load(weights_path))
    return model

def update_input(src_tensor, predicted_tensor):
    src_tensor = torch.cat([src_tensor[1:], predicted_tensor.unsqueeze(0)])
    return src_tensor

def predict_next_frames(model, dataset, n_steps, dump_dir):
    bar = Bar("Step ", max=n_steps)
    src_tensor, tgt_tensor = dataset[len(dataset)]
    src_tensor = update_input(src_tensor, tgt_tensor.squeeze())

    for i in range(1, n_steps + 1):
        # Run prediction
        with torch.no_grad():
            pred = model(src_tensor.unsqueeze(0)).squeeze()
        rescaled_pred = dataset.std * pred + dataset.mean

        # Save predicted image as netcdf file
        dump_path = os.path.join(dump_dir, f"{2100 + i}.nc")
        dataarray = xr.DataArray(np.roll(rescaled_pred.numpy()[::-1], 360, axis=1))
        dataarray.to_netcdf(dump_path)

        # Use prediction as input to next time step
        src_tensor = update_input(src_tensor, pred)
        src_tensor = torch.cat([src_tensor[1:], pred.unsqueeze(0)])
        bar.next()


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}')

    # Run session
    main(args)
