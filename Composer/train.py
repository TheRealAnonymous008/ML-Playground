from model import NoteComposeNet
from dataset import MidiDataset
from torch.utils.data import DataLoader

import torch
import pandas as pd
from tqdm import tqdm

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TrainPipeline: 
    def __init__(self, train_loader, validation_loader, model, loss_fn, optimizer, train_thresh = 10000):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_updates = 0
        self.train_thresh = train_thresh

    def unpack_batch(self, batch):
        b, attn, gt = batch
        
        notes = b['notes'].to(self.model._device)
        notes_gt = gt['notes'].to(self.model._device)

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
                # Tensorboard scalars
                tb_x = epoch_index * len(self.train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)

                # Save intermediate checkpoints
                self.train_updates += 1
                if self.train_updates % self.train_thresh == 0:
                    self.train_updates = 0
                    model_path = 'checkpoints/temps/model'
                    torch.save(self.model.state_dict(), model_path)
                    
        return last_loss
    
    def train(self, epochs):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/composer_{}'.format(timestamp))

        best_vloss = 1_000_000.

        for epoch_number in range(epochs):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)

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

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'checkpoints/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1