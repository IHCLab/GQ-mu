
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from einops import rearrange


class CoreQuantumModule(nn.Module):
    """
    Quantum feature encoder used to generate a compact latent tensor
    for the downstream decoder in QDIP.
    """

    def __init__(self, qubit_num=4, in_channel=24):
        super().__init__()
        assert qubit_num % 2 == 0, "Number of qubits must be even."

        self.qubit_num = qubit_num
        self.in_channel = in_channel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.qml_device = "default.qubit.torch"
            q_device = qml.device(self.qml_device, wires=self.qubit_num, torch_device="cuda")
        else:
            self.qml_device = "default.qubit"
            q_device = qml.device(self.qml_device, wires=self.qubit_num)

        self.measure_set = list(range(qubit_num))
        self.p = nn.Parameter(torch.rand(in_channel, 3, qubit_num) * torch.pi, requires_grad=True)
        self.cp = nn.Parameter(torch.rand(in_channel, qubit_num) * torch.pi, requires_grad=True)

        self.qnode = qml.QNode(
            self.quantum_circuit,
            q_device,
            interface="torch",
            diff_method="backprop",
        )

    def quantum_circuit(self, cp, p):
        for ws in range(self.qubit_num):
            qml.RX(p[:, 0, ws], wires=ws)

        for ws in range(self.qubit_num // 2):
            wss = 2 * ws
            qml.IsingXX(cp[:, ws], wires=[wss, wss + 1])

        for ws in range(self.qubit_num):
            qml.RZ(p[:, 1, ws], wires=ws)

        for ws in range(self.qubit_num // 2):
            wss = 2 * ws + 1
            qml.IsingXX(cp[:, ws + (self.qubit_num // 2)], wires=[wss, (wss + 1) % self.qubit_num])

        for ws in range(self.qubit_num):
            qml.RX(p[:, 2, ws], wires=ws)

        for ws in range(self.qubit_num):
            qml.MultiControlledX(
                control_wires=[ws, (ws + 1) % self.qubit_num],
                wires=(ws + 2) % self.qubit_num,
                control_values="10",
            )

        return [qml.expval(qml.PauliZ(w)) for w in self.measure_set]

    def show_circuit(self):
        fig, ax = qml.draw_mpl(self.quantum_circuit)(self.cp, self.p)
        plt.show()

    def forward(self):
        x = self.qnode(self.cp, self.p)
        x = torch.stack(x).type(torch.float32).permute(1, 0).to(self.device)
        return x


class QDIP(nn.Module):
    """
    QDIP: Quantum Deep Image Prior
    --------
    Core Quatnum Module 
        -> latent tensor of shape [1, 32, 2, 2] when qubit_num=4
        -> transposed-convolution decoder
        -> spectral logits of shape [1, out_channel, H, W]
        -> softmax along channel dimension
        -> flattened abundance / proportion map of shape [out_channel, H*W]
    """

    def __init__(self, device, out_channel, in_channel=4, qubit_num=4, upsample_mode="bicubic"):
        super().__init__()
        assert qubit_num % 2 == 0, "Number of qubits must be even."

        self.device = device
        self.qubit_num = qubit_num
        self.first_in_channel = in_channel
        self.in_channel = 2
        self.scale = 16
        self.slope = 0.1
        self.upsample_mode = upsample_mode

        self.qnn = CoreQuantumModule(
            qubit_num=qubit_num,
            in_channel=self.in_channel * self.scale,
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel * self.scale, self.in_channel * 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 8, self.in_channel * 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 8, self.in_channel * 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 8, self.in_channel * 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=True),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel * 8, self.in_channel * 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 8, self.in_channel * 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 8, self.in_channel * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 4),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 4, self.in_channel * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 4),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel * 4, self.in_channel * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 4),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 4, self.in_channel * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 4),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 4, self.in_channel * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 2),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=True),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel * 2, self.in_channel * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 2),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(self.in_channel * 2, self.in_channel * 1, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.in_channel * 1),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=True),
        )

        self.up_spec = nn.Sequential(
            nn.Conv2d(self.in_channel * 1, out_channel, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self):
        # Quantum features
        x = self.qnn().to(self.device)
        x = x.reshape(1, self.in_channel * self.scale, 2, 2)

        # inverse quantum collapse
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.up_spec(x)

        # softmax-normalized output for sum-to-one and non-negativity
        x = torch.softmax(x, dim=1)
        x = x.squeeze(0)
        x = rearrange(x, "c h w -> c (h w)")
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Toy example
    out_channel = 6
    model = QDIP(device=device, out_channel=out_channel, in_channel=4, qubit_num=4).to(device)

    output = model()

    print("===== QDIP Toy Example =====")
    print("Model name: QDIP")
    print("External input required in forward(): No")
    print(f"Output shape: {tuple(output.shape)}")
    print("Output meaning: [out_channel, H*W] after softmax normalization")
    print("Channel-wise sum at a sample spatial position:",
          output[:, 0].sum().item())
