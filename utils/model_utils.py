from fully_connected_basics.models import FullyConnectedModel
from .config import CONFIGS, DEVICE

def get_model(dataset,
        linear_layer_size, layer_cnt, layer_increase=False, layer_decrease=False,
        batch_norm=False, momentum=0.1, dropout_rate=0.0,
        dropout_increase=False, dropout_decrease=False, dropout_step=0.0):
    layers = []

    for _ in range(layer_cnt):
        linear_layer = {"type": "linear", "size": linear_layer_size}
        batch_norm_layer = {"type": "batch_norm", "momentum": momentum}
        relu_layer = {"type": "relu"}
        dropout_layer = {"type": "dropout", "rate": dropout_rate}

        layers.append(linear_layer)
        if layer_increase:
            linear_layer_size *= 2
        elif layer_decrease:
            linear_layer_size //= 2

        if batch_norm:
            layers.append(batch_norm_layer)

        layers.append(relu_layer)
        
        if dropout_rate > 0:
            layers.append(dropout_layer)
            if dropout_increase:
                dropout_rate += dropout_step
            elif dropout_decrease:
                dropout_rate -= dropout_step
    
    model = FullyConnectedModel(
        input_size=CONFIGS[dataset]['input_size'], 
        num_classes=CONFIGS[dataset]['num_classes'],
        layers=layers).to(DEVICE)
    return model