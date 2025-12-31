import torch.nn as nn

def check_dropout_layers(model_part):
    """
    Finds all Dropout layers in a model and checks if they are in training mode.
    
    Args:
    - model: A PyTorch model (could be nn.Sequential or any nn.Module).
    
    Returns:
    - A list of tuples containing:
      (dropout_layer, is_in_training_mode)
    """

    model_mode = model_part.training
    dropout_status = []

    # Iterate through all named modules in the model
    for name, layer in model_part.named_modules():
        if isinstance(layer, nn.Dropout):
            dropout_status.append((name, layer, layer.training))
            assert layer.training == model_mode, f"Dropout layer {name} is in the wrong mode {layer.training}, should be {model_mode}"
    
    # # Iterate through all modules in the model
    # for layer in model_part.modules():
    #     if isinstance(layer, nn.Dropout):
    #         dropout_status.append((layer, layer.training))
    
    return dropout_status