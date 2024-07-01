import os
import logging
import warnings
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# my module
from dataset.semkitti_trainset import SemanticKITTI
from utils.config import ConfigSemanticKITTI as cfg
from utils.metric import compute_acc, IoUCalculator
from network.SWCFNet import Network
from network.loss_func import compute_loss
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='results', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during training [default: 5]')
parser.add_argument('--val_batch_size', type=int, default=5, help='Batch Size during training [default: 30]')
parser.add_argument('--num_workers', type=int, default=5, help='Number of workers [default: 5]')
FLAGS = parser.parse_args()


class Trainer:
    def __init__(self):

        if not os.path.exists(FLAGS.log_dir):
            os.mkdir(FLAGS.log_dir)
        self.log_dir = FLAGS.log_dir
        log_fname = os.path.join(FLAGS.log_dir, 'log_train.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Trainer")

        train_dataset = SemanticKITTI('training')
        val_dataset = SemanticKITTI('validation')

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=FLAGS.val_batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True
        )
        # Network & Optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Network(cfg)
        self.net.to(device)

        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        # Load module
        self.hightest_val_iou = 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            if checkpoint['best_miou'] is not None:
                self.hightest_val_iou = checkpoint['best_miou']

        # Loss Function
        class_weights = torch.from_numpy(train_dataset.get_class_weight()).float().cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

        # Multiple GPU Training
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        

    def train_one_epoch(self):
        self.net.train()  # set model to training mode
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        Loss = 0
        for batch_idx, batch_data in enumerate(tqdm_loader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(cfg.num_layers):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                else:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            self.optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            end_points = self.net(batch_data)
            loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)
            Loss += loss.item()
            loss.backward()
            self.optimizer.step()
        loss_avg = Loss / len(self.train_loader)
        self.scheduler.step()
        return loss_avg
        
    def train(self):
        
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))

            loss_avg = self.train_one_epoch()
            self.logger.info('**** EVAL EPOCH %03d ****, Loss: %.4f' % (epoch, loss_avg))
            mean_iou = self.validate()
            # Save best checkpoint
            if mean_iou > self.hightest_val_iou:
                self.hightest_val_iou = mean_iou
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
                self.save_checkpoint(checkpoint_file)
            self.logger.info('miou/best_miou:%.1f/%.1f' % (mean_iou*100, self.hightest_val_iou*100))
    def validate(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                torch.cuda.synchronize()
                end_points = self.net(batch_data)

                loss, end_points = compute_loss(end_points, self.train_dataset, self.criterion)

                acc, end_points = compute_acc(end_points)
                iou_calc.add_data(end_points)

        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)
        return mean_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch+1,  # after training one epoch, the start_epoch should be epoch+1
            'best_miou': self.hightest_val_iou,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
