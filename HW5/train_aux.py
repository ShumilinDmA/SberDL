import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import mlflow


def train_epoch(cfg, model, dataloader,  clf_criterion, sparce_criterion, optimizer):
    train_loss = 0
    train_clf_loss = 0
    train_sparce_loss = 0
    correct_train = 0
    size = 0

    model.train()
    for X_cat, X_num, y_batch in tqdm(dataloader):
        x_batch = [X_cat, X_num]
        optimizer.zero_grad()
        output, masks = model(x_batch)

        if cfg.n_output_classes == 1:
            y_float = y_batch.reshape(-1, 1).float()
            clf_loss = clf_criterion(output, y_float)
            predict = (torch.sigmoid(output).detach().numpy() > 0.5).astype(int).ravel()
        else:
            clf_loss = clf_criterion(output, y_batch)
            predict = np.argmax(torch.softmax(output, dim=-1).detach().numpy(), axis=-1)

        sparce_loss = sparce_criterion(masks)
        loss = clf_loss + cfg.lambda_sparce * sparce_loss
        loss.backward()
        optimizer.step()

        train_sparce_loss += sparce_loss.item()
        train_clf_loss += clf_loss.item()
        train_loss += loss.item()
        correct_train += (y_batch.cpu().numpy() == predict).sum()
        size += len(y_batch)

    train_loss = train_loss / size
    train_accuracy = correct_train / size

    return train_loss, train_accuracy, train_clf_loss, train_sparce_loss


def valid_epoch(cfg, model, dataloader,  clf_criterion, sparce_criterion):
    valid_loss = 0
    valid_clf_loss = 0
    valid_sparce_loss = 0
    correct_valid = 0
    size = 0

    model.eval()
    with torch.no_grad():
        for X_cat, X_num, y_batch in tqdm(dataloader):
            x_batch = [X_cat, X_num]
            output, masks = model(x_batch)

            if cfg.n_output_classes == 1:
                y_float = y_batch.reshape(-1, 1).float()
                clf_loss = clf_criterion(output, y_float)
                predict = (torch.sigmoid(output).detach().numpy() > 0.5).astype(int).ravel()
            else:
                clf_loss = clf_criterion(output, y_batch)
                predict = np.argmax(torch.softmax(output, dim=-1).cpu().detach().numpy(), axis=-1)

            sparce_loss = sparce_criterion(masks)
            loss = clf_loss + cfg.lambda_sparce * sparce_loss

            valid_loss += loss.item()
            valid_sparce_loss += sparce_loss.item()
            valid_clf_loss += clf_loss.item()

            correct_valid += (y_batch.cpu().numpy() == predict).sum()
            size += len(y_batch)

        valid_loss = valid_loss / size
        valid_accuracy = correct_valid / size

    return valid_loss, valid_accuracy, valid_clf_loss, valid_sparce_loss


def train_model(cfg,
                model,
                train_dataset: torch.utils.data.Dataset,
                valid_dataset: torch.utils.data.Dataset,
                classification_criterion,
                sparce_criterion,
                optimizer,
                scheduler=None):

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    for epoch in tqdm(range(cfg.epoches)):
        train_loss, train_accuracy, train_clf_loss, train_sparce_loss = train_epoch(cfg, model,
                                                                                    train_loader,
                                                                                    classification_criterion,
                                                                                    sparce_criterion,
                                                                                    optimizer)

        valid_loss, valid_accuracy, valid_clf_loss, valid_sparce_loss = valid_epoch(cfg, model,
                                                                                    valid_loader,
                                                                                    classification_criterion,
                                                                                    sparce_criterion)
        if cfg.scheduler.enable:
            scheduler.step(valid_loss)

        mlflow.log_metric("train/loss", float(train_loss), step=epoch)
        mlflow.log_metric("valid/loss", float(valid_loss), step=epoch)
        mlflow.log_metric("train/accuracy", float(train_accuracy), step=epoch)
        mlflow.log_metric("valid/accuracy", float(valid_accuracy), step=epoch)

        print(f'Epoch: {epoch}, Train loss: {round(train_loss, 4)},  Valid loss: {round(valid_loss, 4)}')
        print(f'Epoch: {epoch}, Train Accuracy: {train_accuracy},  Valid Accuracy: {valid_accuracy}')
        print(f'Train loss clf: {train_clf_loss}, Train loss sparce: {train_sparce_loss}')
        print(f'Valid loss clf: {valid_clf_loss}, Valid loss sparce: {valid_sparce_loss}')
    return "Model is fitted!"


def fit_batch(cfg, model, dataset,
              classification_criterion,
              sparce_criterion,
              optimizer):

    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              shuffle=True, drop_last=True)

    model.train()
    x_cat, x_num, y_batch = next(iter(train_loader))
    x_batch = [x_cat, x_num]

    while True:
        optimizer.zero_grad()
        output, masks = model(x_batch)

        if cfg.n_output_classes == 1:
            y_float = y_batch.reshape(-1, 1).float()
            clf_loss = classification_criterion(output, y_float)
            predict = (torch.sigmoid(output).detach().numpy() > 0.5).astype(int).ravel()
        else:
            clf_loss = classification_criterion(output, y_batch)
            predict = np.argmax(torch.softmax(output, dim=-1).cpu().detach().numpy(), axis=-1)

        sparce_loss = sparce_criterion(masks)
        loss = clf_loss + cfg.lambda_sparce * sparce_loss
        loss.backward()
        optimizer.step()

        train_clf_loss = clf_loss.item()
        train_sparce_loss = sparce_loss.item()
        train_loss = loss.item()

        correct_train = (y_batch.cpu().numpy() == predict).sum()
        train_loss = train_loss / len(x_batch[1])
        train_accuracy = correct_train / len(x_batch[1])

        print(f'Train loss: {round(train_loss, 4)},  Train Accuracy: {train_accuracy},')
        print(f'Train loss clf: {train_clf_loss}, Train loss sparce: {train_sparce_loss}')
