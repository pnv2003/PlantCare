import os
from tempfile import TemporaryDirectory
import time
import torch
from tqdm import tqdm
from regularize import EarlyStopping

def train_model(loaders, model, criterion, optimizer, scheduler, device, num_epochs=25, early_stop_patience=5, early_stop_delta=0.0):
    since = time.time()
    early_stopping = EarlyStopping(patience=early_stop_patience, delta=early_stop_delta)
    stats = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pth')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            stop = False
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Use tqdm to track progress
                for inputs, labels in tqdm(loaders[phase], desc=f'{phase} Epoch {epoch}/{num_epochs - 1}'):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / len(loaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(loaders[phase].dataset)

                if phase == 'valid':
                    scheduler.step(epoch_loss)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'valid':
                    early_stop = early_stopping(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                    
                    if early_stop:
                        print('Early stopping')
                        stop = True
                        break

                stats[f'{phase}_loss'].append(epoch_loss)
                stats[f'{phase}_acc'].append(epoch_acc)

            print()
            if stop:
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, stats

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model