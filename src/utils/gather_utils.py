import torch

def strings_to_tensor(strings, max_string_length=1024):
    """Convert a list of strings to a tensor of ASCII values, padded to a fixed length."""
    tensor = torch.full((len(strings), max_string_length), fill_value=0, dtype=torch.long)
    for i, string in enumerate(strings):
        encoded_string = [ord(c) for c in string][:max_string_length]
        tensor[i, :len(encoded_string)] = torch.tensor(encoded_string, dtype=torch.long)
    return tensor

def tensor_to_strings(tensor):
    """Convert a tensor of ASCII values back to a list of strings."""
    return ["".join(chr(c) for c in row if c > 0) for row in tensor]