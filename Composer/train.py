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
    def __init__(self, midi: MidiDataset, 
                 model, loss_fn, optimizer, 
                 validate = True, batch_size = 32, 
                 train_thresh = 1000, scheduler = None,
                 grad_acc = 16):
        self.midi = midi
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_updates = 0
        self.train_thresh = train_thresh
        self.train_idx = 0
        self.validate = validate
        self.batch_size = batch_size

        self.loader = None 
        self.scheduler : torch.optim.lr_scheduler._LRScheduler = scheduler
        self.grad_acc = grad_acc

    def unpack_batch(self, batch):
        b, attn, gt = batch
        
        notes = b['notes'].to(self.model._device)
        notes_gt = gt['notes'].to(self.model._device)
        attn = attn.to(self.model._device)

        output_logits = self.model.forward(notes)

        return output_logits, notes_gt

    def train_one_epoch(self, epoch_index, tb_writer):
        last_loss = 0
        total_batches = len(self.loader)

        with tqdm(self.loader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_index + 1}")

                with torch.autocast(self.model._device):
                    output_logits, notes_gt = self.unpack_batch(batch)
                    loss = self.loss_fn(output_logits, notes_gt)

                loss /= self.grad_acc
                loss.backward() 

                # Perform gradient accumulation
                if (epoch_index + 1) % self.grad_acc == 0 or epoch_index + 1 == total_batches:
                    # Gather data and report
                    last_loss = loss.detach().item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Progress bar
                tepoch.set_postfix({"train loss": f'{last_loss :.5f}'})

                # Save intermediate checkpoints
                self.train_updates += 1
                if self.train_updates % self.train_thresh == 0:
                    self.train_idx += 1
                    self.train_updates = 0
                    
                    # Tensorboard scalars
                    tb_x = epoch_index * len(self.loader) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)

                    # Save temporary model instance
                    model_path = 'checkpoints/temps/model_' + str(self.train_idx) 
                    torch.save(self.model.state_dict(), model_path)
        
        self.train_idx = 0
        self.train_updates = 0
        
        return last_loss
    
    def train(self, epochs):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/composer_{}'.format(timestamp))

        best_vloss = 1_000_000.

        # Configure data loaders
        self.loader = DataLoader(
            self.midi, 
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True, 
            pin_memory=True,
            pin_memory_device= self.model._device
        )

        for epoch_number in range(epochs):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            self.midi.set_training()
            avg_loss = self.train_one_epoch(epoch_number, writer)

            if self.scheduler is not None: 
                self.scheduler.step()
            self.midi.set_validation()
            if self.validate:
                running_vloss = 0.0
                # Set the model to evaluation mode, disabling dropout and using population
                # statistics for batch normalization.
                self.model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    with tqdm(self.loader, unit="batch") as tepoch:
                        for i, vdata in enumerate(tepoch):
                            tepoch.set_description(f"Epoch {epoch_number + 1}")
                            voutputs, vgt = self.unpack_batch(vdata)

                            vloss = self.loss_fn(voutputs, vgt)
                            running_vloss += vloss.detach().item()
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