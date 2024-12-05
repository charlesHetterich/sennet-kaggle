import time
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch


# Define DICE Coefficient
def dice_coeff(prediction, target):
    
    mask = np.zeros_like(prediction)
    mask[prediction >= 0.5] = 1

    inter = np.sum(mask * target)
    union = np.sum(mask) + np.sum(target)
    epsilon = 1e-6
    result = np.mean(2 * inter / (union + epsilon))
    return result


# Focal Loss
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=0, size_average=None, ignore_index=-100, reduce=None, balance_param=1.0):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        logpt = -F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss

# Train and Test Loop (TODO: Add in Surface dice instead of regular dice)
def train_and_test(model, dataloaders, optimizer, criterion, num_epochs=100, show_images=False):
    since = time.time()
    best_loss = 1e10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = ['epoch', 'training_loss', 'test_loss', 'training_dice_coeff', 'test_dice_coeff']
    train_epoch_losses = []
    test_epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        batchsummary = {a: [0] for a in fieldnames}
        batch_train_loss = 0.0
        batch_test_loss = 0.0

        for phase in ['training', 'test']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            for sample in iter(dataloaders[phase]):
                if show_images:
                    grid_img = make_grid(sample[0])
                    grid_img = grid_img.permute(1,2,0)
                    plt.imshow(grid_img)
                    plt.show()

                inputs = sample[0].to(device)
                masks = sample[1].to(device)

                masks = masks.unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    y_ped = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    batchsummary[f'{phase}_dice_coeff'].append(dice_coeff(y_ped, y_true))
                    
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                        
                        batch_train_loss += loss.item() * sample[0].size(0)
                        
                    else:
                        batch_test_loss += loss.item() * sample[0].size(0)
                        
                if phase == 'training':
                    epoch_train_loss = batch_train_loss / len(dataloaders['training'])
                    train_epoch_losses.append(epoch_train_loss)
                else: 
                    epoch_test_loss = batch_test_loss / len(dataloaders['test'])
                    test_epoch_losses.append(epoch_test_loss)

                batchsummary['epoch'] = epoch

                print('{} Loss: {:.4f}'.format(phase, loss))

            best_loss = np.max(batchsummary['test_dice_coeff'])
            for field in fieldnames[3:]:
                batchsummary[field] = np.mean(batchsummary[field])
            print(
                f'\t\t\t train_dice_coeff: {batchsummary["training_dice_coeff"]}, test_dice_coeff: {batchsummary["test_dice_coeff"]}')

            print('Best dice coefficient: {:4f}'.format(best_loss))

            return model, train_epoch_losses, test_epoch_losses  

