import sys
from itertools import groupby

import torch
import torch.nn as nn
from colorama import Fore
from tqdm import tqdm
from dataloader import *

from torch.utils.tensorboard import SummaryWriter
from model import *


writer = SummaryWriter('log')


model = CRNN().to(gpu)
criterion = nn.CTCLoss(blank=blank_label)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_step = 0

# ================================================ TRAINING MODEL ======================================================
for epoch in range(epochs):
    model.train()
    # ============================================ TRAINING ============================================================
    train_correct = 0
    train_total = 0
    for x_train, y_train,y_len in tqdm(train_loader,
                                 position=0, leave=True):
        # x_train.shape == torch.Size([64, 28, 140])
        batch_size = x_train.shape[0]
        optimizer.zero_grad()
        y_pred = model(x_train.cuda())
        # y_pred.shape == torch.Size([64, 32, 11])
        y_pred = y_pred.permute(1, 0, 2)
        input_lengths = torch.IntTensor(batch_size).fill_(y_pred.size(0))
        loss = criterion(y_pred, y_train, input_lengths, y_len)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train_loss",loss.cpu().item(),train_step)
        train_step += 1

        # max_index.shape == torch.Size([32, 64])
        _, max_index = torch.max(y_pred, dim=2)

        cur_index = 0
        for i in range(batch_size):
            # len(raw_prediction) == 32
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != blank_label])

            if len(prediction) == y_len[i] and torch.all(prediction.eq(y_train[cur_index:cur_index+y_len[i]])):
                train_correct += 1
            train_total += 1
            cur_index += y_len[i]
    print('TRAINING. Correct: ', train_correct, '/',
          train_total, '=', train_correct / train_total)
    writer.add_scalar("train acc",train_correct / train_total,epoch)


    # ============================================ VALIDATION ==========================================================

    model.eval()
    
    val_correct = 0
    val_total = 0
    for x_val, y_val,y_len in tqdm(val_loader,
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        batch_size = x_val.shape[0]
        y_pred = model(x_val.cuda())
        y_pred = y_pred.permute(1, 0, 2)
        input_lengths = torch.IntTensor(batch_size).fill_(y_pred.size(0))
        criterion(y_pred, y_val, input_lengths, y_len)
        _, max_index = torch.max(y_pred, dim=2)
        
        cur_index = 0
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != blank_label])

            if len(prediction) == y_len[i] and torch.all(prediction.eq(y_val[cur_index:cur_index+y_len[i]])):
                val_correct += 1
            val_total += 1
            cur_index += y_len[i]
    print('TESTING. Correct: ', val_correct, '/',
          val_total, '=', val_correct / val_total)

    writer.add_scalar("test acc",val_correct / val_total,epoch)

    torch.save(model.state_dict(),"model.pkl")