import argparse
import os
import torch

from data import PlantVillageDataset
from eval import evaluate, predict
from model import CNN
from train import load_model, save_model, train_model
from vis import show_loss_accuracy

# hyperparameters
batch_size = 64
image_size = 224
num_epochs = 50
lr = 0.001
lr_finetune = 0.0001
weight_decay = 0.0001
early_stop_patience = 5
early_stop_delta = 0.001
# for lr_scheduler of type ReduceLROnPlateau
lr_scheduler_patience = 3
lr_scheduler_factor = 0.1
lr_scheduler_mode = 'min'
# for lr_scheduler of type StepLR
# lr_scheduler_step_size = 7
# lr_scheduler_gamma = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available

# commands:
# python main.py train --model mobilenet_v2 --freeze
# python main.py test --model mobilenet_v2 --freeze

parser = argparse.ArgumentParser(description='Train a model on PlantVillage dataset') 
parser.add_argument('command', type=str, help='Command to execute')
parser.add_argument('--model', type=str, help='Model to use')
parser.add_argument('--freeze', action='store_true', help='Freeze the model (use feature extractor only)')
args = parser.parse_args()

if args.command == 'train':
    print('Training...')
    print(f'Model: {args.model}')
    print(f'Freeze: {args.freeze}')

    PlantVillageDataset.download()
    loaders = PlantVillageDataset.prepare()
    num_classes = len(loaders['train'].dataset.classes)

    model = CNN(args.model, device, args.freeze, num_classes)

    learning_rate = lr if args.freeze else lr_finetune
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_scheduler_mode, factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)
    model, stats = train_model(loaders, model, criterion, optimizer, scheduler, device, num_epochs, early_stop_patience, early_stop_delta)
    print('Training completed!')
    show_loss_accuracy(stats)

    os.makedirs('weights', exist_ok=True)
    save_model(model, f'weights/{args.model}_{'fe' if args.freeze else 'ft'}.pth')

elif args.command == 'test':
    print('Testing...')
    print(f'Model: {args.model}')
    print(f'Freeze: {args.freeze}')

    if not os.path.exists(f'weights/{args.model}.pth'):
        print(f'weights/{args.model}.pth does not exist. Please train the model first.')
        exit()

    PlantVillageDataset.download()
    loaders = PlantVillageDataset.prepare()

    model = CNN(args.model, device, args.freeze)
    load_model(model, f'weights/{args.model}_{'fe' if args.freeze else 'ft'}.pth', device)

    y_true, y_pred = predict(model, loaders, device)
    accuracy, precision, recall, f1_score = evaluate(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

else:
    print(f'Command {args.command} not recognized')









