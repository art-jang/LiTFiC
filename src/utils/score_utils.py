import torch

def average_exp_values(tensors):
    """
    Calculate the average of the exponential values for each tensor in a list.

    Args:
        tensors (list of torch.Tensor): A list of PyTorch tensors.

    Returns:
        list: A list of average exponential values for each tensor.
    """
    averages = []
    
    for tensor in tensors:
        # Apply the exponential function to each element in the tensor
        exp_tensor = torch.exp(tensor)
        
        # Calculate the average of the resulting values
        avg = torch.mean(exp_tensor)
        
        # Append the average to the list
        averages.append(avg.item())  # .item() to convert the tensor to a Python float

    return averages