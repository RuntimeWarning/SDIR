import os
import re
import glob
import torch
import shutil
import numpy as np



class SaveWeights:
    def __init__(self, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.verbose = verbose
        self.best_score = np.inf

    def __call__(self, val_loss, model, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "/log.txt", "a") as f:
            f.write(f'Epoch: {epoch} Loss: {val_loss:.4f}\n')
        self.save_checkpoint(val_loss, model, path, epoch)

    def save_checkpoint(self, val_loss, model, path, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.best_score:.4f} --> {val_loss:.4f}).  Saving model ...')
            
        for filename in glob.iglob(path + '/*'):
            if os.path.isdir(filename):
                num = int(re.findall(r'/(\d+)', filename)[0])
                if num % 5 != 0:
                    shutil.rmtree(filename)

        save_path = path + "/" + str(epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, save_path + "/" + "model_{:.4f}.pth".format(val_loss))
        self.best_score = val_loss