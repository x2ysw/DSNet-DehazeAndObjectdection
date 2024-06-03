import torch

from train.traindehaze import model

torch.save(model.state_dict(), 'dehazing_model_final.pth')