"""
Description : Train CNN to predict next temperature given last three ones

Usage: run_cnn_training.py --root=<path_to_dataset_directory> --o=<path_to_dumping_model_weigths>

Options:
  --root=<path_to_dataset_directory>          Path to dataset directory
  --o=<path_to_model_weigths>                 Path where to save model weights
"""
import os
import logging
from docopt import docopt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from progress.bar import Bar
from src import ConvNet, FieldsDataset


def main(args):
    # Setup experiment
    dataset = FieldsDataset(root=args['--root'])
    logging.info(f"Loading dataset - {len(dataset)} samples")

    model = ConvNet(in_channels=3, out_channels=1, hidden_channels=8)
    logging.info(f"Building model \n {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 10

    # Train model
    logging.info("Fitting model")
    model = train_model(model=model,
                        dataset=dataset,
                        criterion=criterion,
                        optimizer=optimizer,
                        epochs=epochs,
                        dumping_dir=args['--o'])


def train_model(model, dataset, criterion, optimizer, epochs, dumping_dir):
    model.train()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    for i in range(epochs):
        bar = Bar(f"Epoch {i + 1}", max=len(dataset) // 4)
        for batch_idx, (src_tensor, tgt_tensor) in enumerate(iter(dataloader)):
            optimizer.zero_grad()
            output = model(src_tensor)
            loss = criterion(output, tgt_tensor)
            loss.backward()
            optimizer.step()
            bar.suffix = f"Loss {loss.item()}"
            bar.next()
        torch.save(model.state_dict(), os.path.join(dumping_dir, f"epoch_{i}.pt"))
    return model


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}')

    # Run session
    main(args)
