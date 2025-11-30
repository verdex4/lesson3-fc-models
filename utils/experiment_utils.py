from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import plot_training_history, count_parameters
from utils.model_utils import get_model
from utils.visualisation_utils import plot_different_depth_model_accs, plot_accuracy, plot_schemes, plot_weights_distribution, plot_regularization_technique, save_plot
from .config import CONFIGS, DEVICE, LOADERS, PROJECT_ROOT
import numpy as np
import torch

def test_different_depth(dataset: str):
    hidden_layer_sizes = [0, 1, 2, 4, 6]
    linear_layer_sizes = [0, 128, 256, 512, 1024]
    train_loader, test_loader = LOADERS[dataset]

    for layer_cnt, size in zip(hidden_layer_sizes, linear_layer_sizes):
        model = get_model(dataset=dataset, 
                          linear_layer_size=size, layer_cnt=layer_cnt, layer_decrease=True)
        history = train_model(model, train_loader, test_loader, epochs=5, device=DEVICE)
        print(f"{layer_cnt} HIDDEN LAYER MODEL")

        path = f"{PROJECT_ROOT}/plots/depth_experiments/depth_compare/{dataset}"
        name = f"{layer_cnt}_layers"
        plot_training_history(history, path, name)

def test_overfitting_with_different_depth():
    hidden_layer_cnts = [0, 1, 2, 4, 6]
    linear_layer_sizes = [0, 128, 256, 512, 1024]

    for dataset in CONFIGS:
        print(f"DATASET: {dataset.upper()}")
        train_loader, test_loader = LOADERS[dataset]

        for layer_cnt, size in zip(hidden_layer_cnts, linear_layer_sizes):
            simple_model = get_model(dataset=dataset,
                                     linear_layer_size=size, layer_cnt=layer_cnt, layer_decrease=True)
            dropout_bn_model = get_model(dataset=dataset,
                                         linear_layer_size=size, layer_cnt=layer_cnt, layer_decrease=True,
                                         batch_norm=True,
                                         dropout_rate=0.2)
            simple_model_history = train_model(simple_model, 
                                               train_loader, test_loader, epochs=5, device=DEVICE)
            dropout_bn_model_history = train_model(dropout_bn_model,
                                                   train_loader, test_loader, epochs=5, device=DEVICE)
            
            path = f"{PROJECT_ROOT}/plots/depth_experiments/model_compare/{dataset}"
            name = f"{layer_cnt}_layers"
            plot_different_depth_model_accs(simple_model_history, dropout_bn_model_history, layer_cnt,
                                            path, name)

def test_different_width(dataset: str):
    widths = ["narrow", "medium", "wide", "very wide"]
    sizes = [64, 256, 1024, 2048]
    layer_cnt = 3
    train_loader, test_loader = LOADERS[dataset]

    models = []
    for width, size in zip(widths, sizes):
        model = get_model(dataset=dataset,
                          linear_layer_size=size, layer_cnt=layer_cnt, layer_decrease=True)
        models.append(model)
    
    print("PARAMETER ANALYSIS")
    for width, model in zip(widths, models):
        param_cnt = count_parameters(model)
        print(f"{width} model: {param_cnt}")
        
    for width, model in zip(widths, models):
        history = train_model(model, train_loader, test_loader, epochs=5, device=DEVICE)
        path = f"{PROJECT_ROOT}/plots/width_experiments/width_compare/{dataset}"
        name = f"{width}"
        plot_accuracy(history, width, path, name)

def test_different_architectures(dataset, 
                                 schemes: list[str] = ["narrow", "constant", "wide"], 
                                 sizes: list[int] = [64, 128, 256, 512], 
                                 layer_cnt=3):
    train_loader, test_loader = LOADERS[dataset]
    accuracies = dict()

    for scheme in schemes:
        scheme_accs = []
        for size in sizes:
            if scheme == "narrow":
                model = get_model(dataset=dataset,
                                  linear_layer_size=size, layer_cnt=layer_cnt, layer_decrease=True)
            elif scheme == "constant":
                model = get_model(dataset=dataset, 
                                  linear_layer_size=size, layer_cnt=layer_cnt)
            else: # scheme == "wide"
                model = get_model(dataset=dataset,
                                  linear_layer_size=size, layer_cnt=layer_cnt, layer_increase=True)
            history = train_model(model, train_loader, test_loader, epochs=5, device=DEVICE)
            last_acc = history['test_accs'][-1]
            scheme_accs.append(last_acc)
        accuracies[scheme] = scheme_accs
    
    path = f"{PROJECT_ROOT}/plots/depth_experiments/schemes_compare/{dataset}"
    name = "schemes_heatmap"
    plot_schemes(accuracies, sizes, path, name)

def test_regularization_techniques(dataset: str, lr=0.001, weight_decay=0.01):
    techniques = ['simple',
                  'only_dropout01', 'only_dropout03', 'only_dropout05',
                  'only_batch_norm',
                  'batch_norm_dropout01', 'batch_norm_dropout03', 'batch_norm_dropout05']
    train_loader, test_loader = LOADERS[dataset]
    weights = []

    for technique in techniques:
        simple_model, l2_model = get_simple_and_l2_models(dataset, technique)
        
        l2_optimizer = torch.optim.Adam(l2_model.parameters(), lr=lr, weight_decay=weight_decay)
        simple_model_history = train_model(simple_model, train_loader, test_loader, epochs=5, device=DEVICE)
        l2_model_history = train_model(l2_model, train_loader, test_loader, epochs=5, device=DEVICE, 
                                          optimizer=l2_optimizer)

        print(f"TECHNIQUE: {technique.upper()} WITHOUT L2")
        technique_name = f"{technique}_no_l2"
        weights = get_all_weights(simple_model)
        path = f"{PROJECT_ROOT}/plots/regularization_experiments/techniques_compare/{dataset}"
        name = technique_name
        plot_regularization_technique(technique_name, simple_model_history, weights,
                                      path, name)
        
        print(f"TECHNIQUE: {technique.upper()} WITH L2")
        technique_name = f"{technique}_with_l2"
        weights = get_all_weights(l2_model)
        path = f"{PROJECT_ROOT}/plots/regularization_experiments/techniques_compare/{dataset}"
        name = technique_name
        plot_regularization_technique(technique_name, l2_model_history, weights,
                                      path, name)
   
def get_simple_and_l2_models(dataset, technique):
    size, layer_cnt = 512, 3  # лучшие параметры из задания №2

    if "only_dropout" in technique:
        dropout_rate = float(technique[-1]) / 10  # последняя цифра, деленная на 10
        simple_model = get_model(dataset=dataset,
                        linear_layer_size=size, layer_cnt=layer_cnt,
                        dropout_rate=dropout_rate)
        l2_model = get_model(dataset=dataset,
                        linear_layer_size=size, layer_cnt=layer_cnt,
                        dropout_rate=dropout_rate)
        
    elif "only_batch_norm" in technique:
        simple_model = get_model(dataset=dataset,
                        linear_layer_size=size, layer_cnt=layer_cnt,
                        batch_norm=True)
        l2_model = get_model(dataset=dataset,
                        linear_layer_size=size, layer_cnt=layer_cnt,
                        batch_norm=True)
        
    elif "batch_norm" in technique and "dropout" in technique:
        dropout_rate = float(technique[-1]) / 10  # последняя цифра, деленная на 10
        simple_model = get_model(dataset=dataset,
                        linear_layer_size=size, layer_cnt=layer_cnt,
                        batch_norm=True,
                        dropout_rate=dropout_rate)
        l2_model = get_model(dataset=dataset,
                        linear_layer_size=size, layer_cnt=layer_cnt,
                        batch_norm=True,
                        dropout_rate=dropout_rate)
        
    else: # "simple"
        simple_model = get_model(dataset=dataset, linear_layer_size=size, layer_cnt=layer_cnt)
        l2_model = get_model(dataset=dataset, linear_layer_size=size, layer_cnt=layer_cnt)

    return simple_model, l2_model  

def get_all_weights(model):
    all_weights = []
    for param in model.parameters():
        if param.requires_grad: # только обучаемые параметры
            weights = param.data.cpu().numpy().flatten()
            all_weights.extend(weights)
    return np.array(all_weights)

def test_adaptive_techniques(dataset: str):
    techniques = ['increasing_dropout', 'decreasing_dropout',
                  'bn_momentum03', 'bn_momentum05', 'bn_momentum07', 'bn_momentum09',
                  'increasing_dropout_bn_momentum03', 'increasing_dropout_bn_momentum05',
                  'increasing_dropout_bn_momentum07', 'increasing_dropout_bn_momentum09',
                  'decreasing_dropout_bn_momentum03', 'decreasing_dropout_bn_momentum05',
                  'decreasing_dropout_bn_momentum07', 'decreasing_dropout_bn_momentum09']
    
    for technique in techniques:
        model = get_adaptive_model(dataset, technique)
        weights = get_all_weights(model)
        path = f"{PROJECT_ROOT}/plots/regularization_experiments/adaptive_techniques_compare/{dataset}"
        name = technique
        plot_weights_distribution(weights, path, name)

def get_adaptive_model(dataset, technique):
    size, layer_cnt = 512, 3  # лучшие параметры из задания №2
    model = None

    if technique == 'increasing_dropout':
        model = get_model(dataset=dataset,
                            linear_layer_size=size, layer_cnt=layer_cnt,
                            dropout_rate=0.1, dropout_increase=True, dropout_step=0.2)
    
    elif technique == 'decreasing_dropout':
        model = get_model(dataset=dataset,
                            linear_layer_size=size, layer_cnt=layer_cnt,
                            dropout_rate=0.5, dropout_decrease=True, dropout_step=0.2)
    
    elif technique.startswith('bn_momentum'):
        momentum = float(technique[-1]) / 10  # последняя цифра, деленная на 10
        model = get_model(dataset=dataset,
                          linear_layer_size=size, layer_cnt=layer_cnt,
                          batch_norm=True, momentum=momentum)
    
    elif technique.startswith('increasing_dropout_bn_momentum'):
        momentum = float(technique[-1]) / 10  # последняя цифра, деленная на 10
        model = get_model(dataset=dataset,
                          linear_layer_size=size, layer_cnt=layer_cnt,
                          batch_norm=True, momentum=momentum,
                          dropout_rate=0.1, dropout_increase=True, dropout_step=0.2)
    
    elif technique.startswith('decreasing_dropout_bn_momentum'):
        momentum = float(technique[-1]) / 10  # последняя цифра, деленная на 10
        model = get_model(dataset=dataset,
                          linear_layer_size=size, layer_cnt=layer_cnt,
                          batch_norm=True, momentum=momentum,
                          dropout_rate=0.5, dropout_decrease=True, dropout_step=0.2)
    
    return model

