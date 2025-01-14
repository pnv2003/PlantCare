from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import torch

def predict(model, loaders, device):

    batch_size = loaders['test'].batch_size
    model.eval()
    y_true = []
    y_pred = []
    inference_times = []

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time.record()
            outputs = model(inputs)
            end_time.record()
            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time))

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(f'Average inference time: {sum(inference_times) / (len(inference_times) * batch_size) / 1000:.4f} seconds')
    return y_true, y_pred

def evaluate(y_true, y_pred):

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, precision, recall, f1_score

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', cmap='Blues'):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap=cmap)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def show_false_prediction(images, labels, y_true, y_pred, grid_size=(6, 6)):

    false_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]

    if len(false_idx) > grid_size[0] * grid_size[1]:
        print(f"Number of false predictions {len(false_idx)} exceeds the grid size {grid_size[0]}x{grid_size[1]}")
        return

    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        idx = false_idx[i]
        ax.imshow(images[idx].permute(1, 2, 0))
        ax.axis('off')
        ax.set_title(f'True: {labels[y_true[idx]]} Pred: {labels[y_pred[idx]]}')

    plt.tight_layout()
    plt.show()