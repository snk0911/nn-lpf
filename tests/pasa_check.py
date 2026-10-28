"""Minimal PASA equivalence check: original vs. stride-aware unfold.

Run: python pasa_quick_check.py
Requires: PyTorch only. No training or dataset.
"""
import torch
from torch import nn
from torch.nn import functional as F

class OriginalPASA(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super().__init__()
        if pad_type != 'reflect':
            raise ValueError('This standalone test supports reflect padding only')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size,
                              kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]


class OptimizedPASA(OriginalPASA):
    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        sigma = sigma[:, :, ::self.stride, ::self.stride]  # CHANGED: sampled attention positions

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,_,_ = x.shape  # CHANGED: keep sampled h, w
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, stride=self.stride).reshape((n,c,self.kernel_size*self.kernel_size,h*w))  # CHANGED: sampled patches

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x  # CHANGED: already downsampled



if __name__ == "__main__":
    # Use the GPU when available; otherwise run the same test on the CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check both PASA kernel sizes used in the experiments.
    for k in (3, 5):  # Remove 5 if you use PASA-3 only
        # Fix the random seed so this test uses reproducible parameters and inputs.
        torch.manual_seed(42)

        # Create the original and memory-optimized PASA layers in training mode.
        # Stride 2 is the case where the optimization skips unused patch positions.
        original = OriginalPASA(16, k, stride=2, group=8).to(device).train()
        optimized = OptimizedPASA(16, k, stride=2, group=8).to(device).train()

        # Copy the original weights and BatchNorm state into the optimized layer.
        # Both layers must start identically for a meaningful comparison.
        optimized.load_state_dict(original.state_dict())

        # Generate one small synthetic feature map: (batch, channels, height, width).
        # Make independent copies so we can compare gradients for the same input.
        x = torch.randn(2, 16, 16, 16, device=device)
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()

        # FORWARD CHECK: both layers should produce (almost) identical outputs.
        # Small differences are allowed because floating-point arithmetic rounds.
        y1, y2 = original(x1), optimized(x2)
        torch.testing.assert_close(y1, y2, rtol=1e-4, atol=1e-5)

        # BACKWARD CHECK: backpropagate the same simple test loss through both.
        # First, compare gradients with respect to the input feature map.
        y1.square().mean().backward()
        y2.square().mean().backward()
        torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-5)

        # Compare gradients of trainable parameters (convolution and BatchNorm).
        # This checks that both implementations produce equivalent training signals.
        for name, p1 in original.named_parameters():
            p2 = dict(optimized.named_parameters())[name]
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5)

        # Compare non-trainable state, including BatchNorm running statistics.
        # These should be unchanged because attention is sampled AFTER BatchNorm.
        for name, b1 in original.named_buffers():
            b2 = dict(optimized.named_buffers())[name]
            torch.testing.assert_close(b1, b2, rtol=1e-4, atol=1e-5)

        # This line is only reached when all assertions for this kernel pass.
        print(f"PASA K={k}: PASS (forward, backward, BatchNorm)")

    print(f"All checks passed on {device} (FP32).")
