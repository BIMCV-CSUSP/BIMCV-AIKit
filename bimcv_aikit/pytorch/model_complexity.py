from ptflops import get_model_complexity_info

def count_parameters_pytorch(model):
    """
    Given a PyTorch model, return the total number of trainable parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def calculate_model_complexity(model, input_size):
    
    #Calculate Number of Flops
    flops, params_1 = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    print('FLOPs: ' + str(flops))
    print('Params: ' + str(params_1))
    
    return flops