import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetModel(nn.Module):
    def __init__(self, state_dim=16, one_hot_depth=16, h1_dim=5000, resnet_dim=1000,
                 num_resnet_blocks=4, out_dim=1, batch_norm=True):
        super().__init__()
        self.one_hot_depth = one_hot_depth
        self.state_dim = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm

        if one_hot_depth > 0:
            self.fc1 = nn.Linear(state_dim * one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(state_dim, h1_dim)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        for _ in range(num_resnet_blocks):
            if batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)

        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)
            x = F.relu(x + res_inp)

        x = self.fc_out(x)
        return x


class OfficialModelWrapper:
    def __init__(self, model, device):
        self._model = model
        self._device = device

    def predict_j(self, grid):
        """grid: (16,) numpy int64 → Python float."""
        x = torch.from_numpy(np.asarray(grid, dtype=np.int64)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            val = self._model(x).item()
        return val

    def predict_j_batch(self, grids):
        """grids: (B, 16) numpy int64 → (B,) tensor on device."""
        x = torch.from_numpy(np.asarray(grids, dtype=np.int64)).to(self._device)
        with torch.inference_mode():
            val = self._model(x).squeeze(-1)
        return val


def load_official_model(model_path, use_gpu=False):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = ResnetModel().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # 剥离 DataParallel 的 module. 前缀
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return OfficialModelWrapper(model, device)
