from model import NoteComposeNet
from dataset import MidiDataset
from torch.utils.data import DataLoader

import torch
import pandas as pd
from tqdm import tqdm

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import dataset

class TrainPipeline: 
    def __init__(self, train_midi : MidiDataset , vali_midi : MidiDataset, model, loss_fn, optimizer, validate = True, batch_size = 32, train_thresh = 2000):
        self.train_midi = train_midi
        self.vali_midi = vali_midi
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_updates = 0
        self.train_thresh = train_thresh
        self.train_idx = 0
        self.validate = validate
        self.batch_size = batch_size

        self.train_loader = None 
        self.vali_loader = None

    def unpack_batch(self, batch):
        b, attn, gt = batch
        
        notes = b['notes'].to(self.model._device)
        notes_gt = gt['notes'].to(self.model._device)
        attn = attn.to(self.model._device)

        output_logits = self.model.forward(notes)

        return output_logits, notes_gt

    def train_one_epoch(self, epoch_index, tb_writer):
        total_loss = 0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_index + 1}")

                self.optimizer.zero_grad()
                
                output_logits, notes_gt = self.unpack_batch(batch)
                
                loss = self.loss_fn(output_logits, notes_gt)
                loss.backward() 

                self.optimizer.step()
                
                # Gather data and report
                total_loss += loss.item()
                last_loss = loss.item()

                # Progress bar
                tepoch.set_postfix({"train loss": f'{last_loss :.5f}'})

                # Save intermediate checkpoints
                self.train_updates += 1
                if self.train_updates % self.train_thresh == 0:
                    self.train_idx += 1
                    self.train_updates = 0
                    
                    # Tensorboard scalars
                    tb_x = epoch_index * len(self.train_loader) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)

                    # Save temporary model instance
                    model_path = 'checkpoints/temps/model_' + str(self.train_idx) 
                    torch.save(self.model.state_dict(), model_path)
        
        self.train_idx = 0
        self.train_updates = 0
        
        return last_loss
    
    def train(self, epochs, start_length = 16):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/composer_{}'.format(timestamp))

        best_vloss = 1_000_000.

        self.train_midi._start_length = start_length
        self.vali_midi._start_length = start_length

        for epoch_number in range(epochs):
            # Configure data loaders
            self.train_loader = DataLoader(
                self.train_midi, 
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=True, 
            )
            self.validation_loader = DataLoader(
                self.vali_midi, 
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=True
            )

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)

            if self.validate:
                running_vloss = 0.0
                # Set the model to evaluation mode, disabling dropout and using population
                # statistics for batch normalization.
                self.model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    with tqdm(self.validation_loader, unit="batch") as tepoch:
                        for i, vdata in enumerate(tepoch):
                            tepoch.set_description(f"Epoch {epoch_number + 1}")
                            voutputs, vgt = self.unpack_batch(vdata)

                            vloss = self.loss_fn(voutputs, vgt)
                            running_vloss += vloss.item()
                            avg_vloss = running_vloss / (i + 1)

                            tepoch.set_postfix({"val loss": f'{avg_vloss:.5f}'})
                
                avg_vloss = running_vloss / (i + 1)

                # Log the running loss averaged per batch
                # for both training and validation
                writer.add_scalars('Training vs. Validation Loss',
                                { 'Training' : avg_loss, 'Validation' : avg_vloss },
                                epoch_number + 1)
                writer.flush()

            model_path = 'checkpoints/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(self.model.state_dict(), model_path)

            epoch_number += 1