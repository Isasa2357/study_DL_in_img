# %% [markdown]
# # データセットインポート

# %%
# torch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# numpy
import numpy as np

# other
import random
from tqdm import tqdm
from enum import Enum

# %% [markdown]
# # ランダムシード固定

# %%
is_fixing = True
if is_fixing:
    torch.manual_seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.default_rng(seed=42)
    random.seed(42)

# %% [markdown]
# # ハイパーパラメータ定義

# %%
batch_size = 64
lr = 0.01

# %% [markdown]
# # モデル定義

# %%
class MyBottleneckBlock(nn.Module):
    '''
    ResNetを構成するBlockクラス
    '''

    def __init__(self, 
                 in_channel: int, 
                 out_channel: int, 
                 halve_spatial: bool=False, 
                 first_conv_compression_rate: int=1):

        super().__init__()

        # 1x1 conv2d: compressing channel
        conv1_in_channel = in_channel
        conv1_out_channel = int(in_channel / first_conv_compression_rate)
        self._conv1 = nn.Conv2d(
            in_channels=conv1_in_channel, 
            out_channels=conv1_out_channel, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self._bn1 = nn.BatchNorm2d(conv1_out_channel)

        # 3x3 conv2d: if need, compressing image size
        conv2_in_channel = conv1_out_channel
        conv2_out_channel = conv2_in_channel
        conv2_kernel_size = 3
        conv2_stride = 1 if not halve_spatial else 2
        conv2_padding = 1
        self._conv2 = nn.Conv2d(
            in_channels=conv2_in_channel, 
            out_channels=conv2_out_channel, 
            kernel_size=conv2_kernel_size, 
            stride=conv2_stride, 
            padding=conv2_padding
        )
        self._bn2 = nn.BatchNorm2d(conv2_out_channel)

        # 1x1 conv2d: expand channel.
        conv3_in_channel = conv2_out_channel
        conv3_out_channel = out_channel
        conv3_kernel_size = 1
        conv3_stride = 1
        conv3_padding = 0
        self._conv3 = nn.Conv2d(
            in_channels=conv3_in_channel, 
            out_channels=conv3_out_channel, 
            kernel_size=conv3_kernel_size, 
            stride=conv3_stride, 
            padding=conv3_padding
        )
        self._bn3 = nn.BatchNorm2d(conv3_out_channel)

        # identity: if chaning image size or channel, adjusting input tensor
        self._identity_conv: nn.Module | None = None
        self._identity_bn: nn.Module | None = None
        if (in_channel != out_channel) or halve_spatial:
            self._identity_conv = nn.Conv2d(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=1, 
                stride=1 if not halve_spatial else 2, 
                padding=0
            )

            self._identity_bn = nn.BatchNorm2d(out_channel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity: torch.Tensor
        if isinstance(self._identity_conv, nn.Module) and isinstance(self._identity_bn, nn.Module):
            identity = self._identity_conv.forward(x)
            identity = self._identity_bn(identity)
        else:
            identity = x.clone()

        x = self._conv1.forward(x)
        x = self._bn1.forward(x)
        x = F.relu(x)


        x = self._conv2.forward(x)
        x = self._bn2.forward(x)
        x = F.relu(x)

        x = self._conv3.forward(x)
        x = self._bn3.forward(x)

        x = x + identity
        x = F.relu(x)

        return x

class MyResNet(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, first_conv_out_channel: int=64):
        '''
        param:
            in_shape: H, W, C
            out_size: output size of MyResNet
        '''
        super().__init__()

        # [3, 224, 224] -> [64, 112, 112]
        self._conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, 
                out_channels=first_conv_out_channel, 
                kernel_size=7, 
                stride=2, 
                padding=3
            ), 
            nn.BatchNorm2d(first_conv_out_channel), 
            nn.ReLU()
        )

        # [64, 112, 112] -> [256, 56, 56]
        conv2_out_channel = first_conv_out_channel*4
        self._conv2 = nn.Sequential(
            # [64, 112, 112] -> [56, 56, 64]
            nn.MaxPool2d(
                kernel_size=3, 
                stride=2, 
                padding=1
            ), 
            # 1x1: [64, 56, 56] -> [64, 56, 56]
            # 3x3: [64, 56, 56] -> [64, 56, 56]
            # 1x1: [64, 56, 56] -> [256, 56, 56]
            MyBottleneckBlock(first_conv_out_channel, conv2_out_channel), 
            # 1x1: [256, 56, 56] -> [64, 56, 56]
            # 3x3: [64, 56, 56] -> [64, 56, 56]
            # 1x1: [64, 56, 56] -> [256, 56, 56]
            MyBottleneckBlock(conv2_out_channel, conv2_out_channel, first_conv_compression_rate=4), 
            # 1x1: [256, 56, 56] -> [64, 56, 56]
            # 3x3: [64, 56, 56] -> [64, 56, 56]
            # 1x1: [64, 56, 56] -> [256, 56, 56]
            MyBottleneckBlock(conv2_out_channel, conv2_out_channel, first_conv_compression_rate=4)
        )

        # [256, 56, 56] -> [512, 28, 28]
        conv3_out_channel = conv2_out_channel * 2
        self._conv3 = nn.Sequential(
            # 1x1: [256, 56, 56] -> [128, 56, 56]
            # 3x3: [128, 56, 56] -> [128, 28, 28]
            # 1x1: [128, 28, 28] -> [512, 28, 28]
            MyBottleneckBlock(conv2_out_channel, conv3_out_channel, halve_spatial=True, first_conv_compression_rate=2), 
            # 1x1: [512, 28, 28] -> [128, 28, 28]
            # 3x3: [128, 28, 28] -> [128, 28, 28]
            # 1x1: [128, 28, 28] -> [512, 28, 28]
            MyBottleneckBlock(conv3_out_channel, conv3_out_channel, first_conv_compression_rate=4),
            # 1x1: [512, 28, 28] -> [128, 28, 28]
            # 3x3: [128, 28, 28] -> [128, 28, 28]
            # 1x1: [128, 28, 28] -> [512, 28, 28]
            MyBottleneckBlock(conv3_out_channel, conv3_out_channel, first_conv_compression_rate=4),
            # 1x1: [512, 28, 28] -> [128, 28, 28]
            # 3x3: [128, 28, 28] -> [128, 28, 28]
            # 1x1: [128, 28, 28] -> [512, 28, 28]
            MyBottleneckBlock(conv3_out_channel, conv3_out_channel, first_conv_compression_rate=4)
        )
        
        # [512, 28, 28] -> [1024, 14, 14]
        conv4_out_channel = conv3_out_channel * 2
        self._conv4 = nn.Sequential(
            # 1x1: [512, 28, 28] -> [256, 28, 28]
            # 3x3: [256, 28, 28] -> [256, 14, 14]
            # 1x1: [256, 14, 14] -> [1024, 14, 14]
            MyBottleneckBlock(conv3_out_channel, conv4_out_channel, halve_spatial=True, first_conv_compression_rate=2), 
            # 1x1: [1024, 14, 14] -> [256, 14, 14]
            # 3x3: [256, 14, 14] -> [256, 14, 14]
            # 1x1: [256, 14, 14] -> [1024, 14, 14]
            MyBottleneckBlock(conv4_out_channel, conv4_out_channel, first_conv_compression_rate=4), 
            # 1x1: [1024, 14, 14] -> [256, 14, 14]
            # 3x3: [256, 14, 14] -> [256, 14, 14]
            # 1x1: [256, 14, 14] -> [1024, 14, 14]
            MyBottleneckBlock(conv4_out_channel, conv4_out_channel, first_conv_compression_rate=4), 
            # 1x1: [1024, 14, 14] -> [256, 14, 14]
            # 3x3: [256, 14, 14] -> [256, 14, 14]
            # 1x1: [256, 14, 14] -> [1024, 14, 14]
            MyBottleneckBlock(conv4_out_channel, conv4_out_channel, first_conv_compression_rate=4), 
            # 1x1: [1024, 14, 14] -> [256, 14, 14]
            # 3x3: [256, 14, 14] -> [256, 14, 14]
            # 1x1: [256, 14, 14] -> [1024, 14, 14]
            MyBottleneckBlock(conv4_out_channel, conv4_out_channel, first_conv_compression_rate=4), 
            # 1x1: [1024, 14, 14] -> [256, 14, 14]
            # 3x3: [256, 14, 14] -> [256, 14, 14]
            # 1x1: [256, 14, 14] -> [1024, 14, 14]
            MyBottleneckBlock(conv4_out_channel, conv4_out_channel, first_conv_compression_rate=4)
        )

        # [1024, 14, 14] -> [2056, 7, 7]
        conv5_out_channel = conv4_out_channel * 2
        self._conv5 = nn.Sequential(
            # 1x1: [1024, 14, 14] -> [512, 14, 14]
            # 3x3: [512, 14, 14] -> [512, 7, 7]
            # 1x1: [512, 7, 7] -> [2056, 7, 7]
            MyBottleneckBlock(conv4_out_channel, conv5_out_channel, halve_spatial=True, first_conv_compression_rate=2), 
            # 1x1: [2056, 7, 7] -> [512, 7, 7]
            # 3x3: [512, 7, 7] -> [512, 7, 7]
            # 1x1: [512, 7, 7] -> [2056, 7, 7]
            MyBottleneckBlock(conv5_out_channel, conv5_out_channel, first_conv_compression_rate=4), 
            # 1x1: [2056, 7, 7] -> [512, 7, 7]
            # 3x3: [512, 7, 7] -> [512, 7, 7]
            # 1x1: [512, 7, 7] -> [2056, 7, 7]
            MyBottleneckBlock(conv5_out_channel, conv5_out_channel, first_conv_compression_rate=4)
        )

        self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._fc = nn.Linear(conv5_out_channel, out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1.forward(x)
        # print(f"conv1 out shape: {x.shape}")
        x = self._conv2.forward(x)

        x = self._conv3.forward(x)
        # print(f"conv3 out shape: {x.shape}")

        x = self._conv4.forward(x)
        # print(f"conv4 out shape: {x.shape}")

        x = self._conv5.forward(x)
        # print(f"conv5 out shape: {x.shape}")

        x = self._avgpool.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self._fc.forward(x)

        return x


# %% [markdown]
# # データセットダウンロード + データローダー作成

# %% [markdown]
# ## データセットはCIFAR-10 or CIFAR-100を使用．フラグにより切り替え

# %%
class CIFAR_Dataset(Enum):
    CIFAR10 = 0
    CIFAR100 = 1

# %%
used_dataset = CIFAR_Dataset.CIFAR10

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize(size=(224, 224))
])

if used_dataset == CIFAR_Dataset.CIFAR10:
    train_dataset = torchvision.datasets.CIFAR10(
        root="./dataset", 
        train=True, 
        transform=transform, 
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./dataset", 
        train=False, 
        transform=transform, 
        download=True
    )
elif used_dataset == CIFAR_Dataset == CIFAR_Dataset.CIFAR100:
    train_dataset = torchvision.datasets.CIFAR100(
        root="./dataset", 
        train=True, 
        transform=transform, 
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./dataset", 
        train=False, 
        transform=transform, 
        download=True
    )
else:
    used_dataset = CIFAR_Dataset.CIFAR10

    train_dataset = torchvision.datasets.CIFAR10(
        root="./dataset", 
        train=True, 
        transform=transform, 
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./dataset", 
        train=False, 
        transform=transform, 
        download=True
    )

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_ds = torchvision.datasets.Food101(
    root="./dataset",
    split="train",
    transform=train_tf,
    download=True,
)

test_ds = torchvision.datasets.Food101(
    root="./dataset",
    split="test",
    transform=test_tf,
    download=True,
)


# %% [markdown]
# ## DataLoader作成

# %%


# %% [markdown]
# # 学習 & 評価 関数定義

# %%
def train_one_epoch(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> tuple[float, float]:
    
    model.train()

    total_loss: float = 0.0
    num_samples: int = 0
    total_correct: int = 0

    for i, (bimgs, blabels) in tqdm(enumerate(train_loader), position=1):
        bimgs: torch.Tensor = bimgs.to(device)
        blabels: torch.Tensor = blabels.to(device)

        # 推論
        outputs: torch.Tensor = model.forward(bimgs)
        preds: torch.Tensor = torch.argmax(outputs, dim=1)

        # 損失計算
        loss: torch.Tensor = criterion.forward(outputs, blabels)

        # パラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 集計
        num_samples += bimgs.shape[0]
        total_loss += loss.item() * bimgs.shape[0]
        total_correct += int((preds == blabels).sum().item())
    
    acc = total_correct / num_samples
    ave_loss = total_loss / num_samples

    return acc, ave_loss

def eval_model(
    model: nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> tuple[float, float]:
    
    model.eval()

    num_sample: int = 0
    total_loss: float = 0.0
    total_correct: int = 0

    with torch.no_grad():
        for i, (bimgs, blabels) in tqdm(enumerate(test_loader), position=1):
            bimgs: torch.Tensor = bimgs.to(device)
            blabels: torch.Tensor = blabels.to(device)

            # 推論
            outputs: torch.Tensor = model.forward(bimgs)
            preds: torch.Tensor = torch.argmax(outputs, dim=1)

            # 集計
            loss: torch.Tensor = criterion.forward(outputs, blabels)
            num_sample += bimgs.shape[0]
            total_loss += loss.item() * bimgs.shape[0]
            total_correct += int((preds == blabels).sum().item())
    
    acc = total_correct / num_sample
    ave_loss = total_loss / num_sample

    return acc, ave_loss

def train_loop(
    epochs: int, 
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    test_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> None:
    
    for epoch in tqdm(range(epochs), position=0):
        tqdm.write(f"epoch: {epoch + 1}")
        train_acc, train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device
        )

        tqdm.write(f"train acc: {train_acc}, loss: {train_loss}")

        e_acc, e_loss = eval_model(
            model, 
            test_loader, 
            criterion, 
            device
        )

        tqdm.write(f"eval acc: {e_acc}, loss: {e_loss}")



# %% [markdown]
# # 学習

# %%

def main():
    train_loader = torch.utils.data.DataLoader(
        # dataset=train_dataset, 
        dataset=train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        # dataset=test_dataset, 
        dataset=test_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MyResNet(3, 101).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loop(
        200, 
        model, 
        train_loader,
        test_loader, 
        optimizer, 
        criterion, 
        device
    )

    torch.save(model, "ResNet_CIFAR10.pth")

if __name__ == "__main__":
    main()
