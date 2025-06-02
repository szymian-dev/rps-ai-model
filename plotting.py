import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
    
def plot_accuracy(history):
    acc = history['acc']
    val_acc = history['val_acc']
    
    if acc is None or val_acc is None:
        raise Exception('accuracy or val_accuracy not found in history')
    
    plt.clf()
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo--', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
def plot_loss(history):
    loss = history['loss']
    val_loss = history['val_loss']
    
    if loss is None or val_loss is None:
        raise Exception('loss or val_loss not found in history')
    
    plt.clf()
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'ro--', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()

def plot_accuracy_and_loss(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']
    
    if loss is None or val_loss is None or acc is None or val_acc is None:
        raise Exception('loss, val_loss, accuracy or val_accuracy not found in history')
    
    assert len(loss) == len(acc) == len(val_loss) == len(val_acc)
    
    plt.clf()
    epochs = range(len(loss))
    fig, axs = plt.subplots(2, 1, figsize=(25, 10)) 
    axs[0].plot(epochs, acc, 'bo--', label='Training accuracy')
    axs[0].plot(epochs, val_acc, 'b-', label='Validation accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Train', 'Validation'])
    
    axs[1].plot(epochs, loss, 'ro--', label='Training loss')
    axs[1].plot(epochs, val_loss, 'r-', label='Validation loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['Train', 'Validation'])
    plt.show()
    
def plot_history_csv(csv_file, title=None):
    history = pd.read_csv(csv_file)
    
    required_columns = ['acc', 'val_acc', 'loss', 'val_loss']
    for col in required_columns:
        if col not in history.columns:
            raise Exception(f"Column '{col}' not found in history")

    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']
    
    assert len(loss) == len(acc) == len(val_loss) == len(val_acc)

    plt.clf()
    epochs = range(1, len(loss) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(25, 10)) 
    

    axs[0].plot(epochs, acc, 'bo--', label='Training accuracy')
    axs[0].plot(epochs, val_acc, 'b-', label='Validation accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Train', 'Validation'])
    
    axs[1].plot(epochs, loss, 'ro--', label='Training loss')
    axs[1].plot(epochs, val_loss, 'r-', label='Validation loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['Train', 'Validation'])
    
    if title:
        plt.suptitle(title)
    plt.show()
        
        
def plot_grid_from_generator(generator, rows, cols, cmap='viridis'):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  
    axes = axes.flatten()  

    for i in range(rows * cols):
        batch = next(generator) 
        images = batch[0]                               # 0 - images, 1 - labels
        image_augmented = images[0]                     # (batch_size, height, width, channels) -> (height, width, channels)
        axes[i].imshow(image_augmented, cmap=cmap)
        axes[i].axis('off')  

    plt.tight_layout() 
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
