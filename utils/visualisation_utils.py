import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def save_plot(path, name):
    """Сохраняет график по указанному пути"""
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/{name}.png', dpi=300, bbox_inches='tight')

def plot_different_depth_model_accs(simple_model_history, dropout_bn_model_history, layer_cnt, 
                                    save_path=None, name=None):
    """Визуализирует и сохраняет train/test accuracy для двух моделей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(simple_model_history['train_accs'], label='Train Acc')
    ax1.plot(simple_model_history['test_accs'], label='Test Acc')
    ax1.set_title(f'Simple model acc ({layer_cnt} hidden layers)')
    ax1.legend()
    
    ax2.plot(dropout_bn_model_history['train_accs'], label='Train Acc')
    ax2.plot(dropout_bn_model_history['test_accs'], label='Test Acc')
    ax2.set_title(f'Dropout, bn model acc ({layer_cnt} hidden layers)')
    ax2.legend()
    
    plt.tight_layout()
    if save_path and name:
        save_plot(save_path, name)
    plt.show()

def plot_accuracy(history, width: str, save_path=None, name=None):
    """Визуализирует и сохраняет точность во время обучения"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(history['train_accs'], label='Train Acc')
    ax.plot(history['test_accs'], label='Test Acc')
    ax.set_title(f'Acc (width: {width})')
    ax.legend()
    
    plt.tight_layout()
    if save_path and name:
        save_plot(save_path, name)
    plt.show()

def plot_schemes(data, sizes: list, save_path=None, name=None):
    df = pd.DataFrame(data, index=sizes)
    sns.heatmap(df)

    plt.tight_layout()
    if save_path and name:
        save_plot(save_path, name)
    plt.show()

def plot_weights_distribution(weights, save_path=None, name=None):
    plt.figure(figsize=(7, 3))
    plt.hist(weights, bins=150)
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.xlim(-0.2, 0.2)

    plt.tight_layout()
    if save_path and name:
        save_plot(save_path, name)
    plt.show()

def plot_regularization_technique(technique_name, history, weights, save_path=None, name=None):
    """Объединяет графики обучения и распределение весов"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accs'], label='Train Acc')
    plt.plot(history['test_accs'], label='Test Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, (3, 4))
    plt.hist(weights, bins=150)
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.xlim(-0.2, 0.2)
    
    plt.suptitle(f'Technique: {technique_name}')
    plt.tight_layout()
    
    if save_path and name:
        save_plot(save_path, name)
    plt.show()
