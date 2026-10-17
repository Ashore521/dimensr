import torch

# Tests use tiny recurrent matrices. Avoid CPU thread startup dominating runtime.
torch.set_num_threads(1)
