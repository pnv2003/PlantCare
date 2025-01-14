from matplotlib import pyplot as plt
import seaborn as sns

def show_class_distribution(dataset):

    classes = dataset.classes
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    targets = [idx_to_class[t] for t in dataset.targets]

    plt.figure(figsize=(10, 5))
    sns.countplot(targets)
    plt.xticks(rotation=90)
    plt.title('Class Distribution')
    plt.show()

def show_images(images, labels, grid_size=(6, 6)):

    if len(images) > grid_size[0] * grid_size[1]:
        print(f"Number of images {len(images)} exceeds the grid size {grid_size[0]}x{grid_size[1]}")
        return
    
    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.axis('off')
        ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()

def show_loss_accuracy(stats):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(stats['train_loss'], label='train')
    ax[0].plot(stats['valid_loss'], label='valid')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(stats['train_acc'], label='train')
    ax[1].plot(stats['valid_acc'], label='valid')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()