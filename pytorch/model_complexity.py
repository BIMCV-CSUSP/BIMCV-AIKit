def count_parameters_pytorch(model):
    """
    Given a PyTorch model, return the total number of trainable parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
