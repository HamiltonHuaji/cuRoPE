# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    import curope._C as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import _C as _kernels # run `python setup.py build_ext --inplace`


class cuRoPE2D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # uncomment this if inplace doesn't work
        tokens = tokens.clone().contiguous()
        _kernels.rope_2d(tokens, positions, base, F0)
        # ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.rope_2d(grad_res, positions, base, -F0)
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions):
        """
        Apply 2D RoPE to the input tokens.
        Args:
            tokens (torch.Tensor): Input tokens of shape (B,H,N,D).
            positions (torch.Tensor): Position encodings of shape (B,N,2).
        Example:
            import torch; from curope import cuRoPE2D; rope = cuRoPE2D()
            tokens = rope(tokens=torch.randn((1,4,13,8)).cuda(), positions=torch.zeros((1,13,2)).long().cuda())

        Returns:
            torch.Tensor: Tokens with RoPE applied, of shape (B,H,N,D).
        """
        cuRoPE2D_func.apply( tokens.transpose(1,2), positions, self.base, self.F0 )
        return tokens