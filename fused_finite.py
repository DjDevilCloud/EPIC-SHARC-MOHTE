# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional out-of-place CUDA finite repair on the PyTorch current stream.

Healthy values are copied; inputs are never mutated. Counts stay on the GPU.
CuPy is imported only when this backend is explicitly selected.
"""
from functools import lru_cache
import os
from pathlib import Path

import torch


class _TorchStream:
    def __init__(self, stream):
        self.stream = stream

    def __cuda_stream__(self):
        return 0, self.stream.cuda_stream


@lru_cache(maxsize=64)
def _kernels(dtype, sizes):
    # Keep NVRTC artifacts inside the workspace, including on restricted hosts.
    os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).parent / "review_artifacts" / "cupy_cache"))
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("training_finite_guard_backend=cuda requires CuPy (cupy-cuda12x for CUDA 12)") from exc
    types = {torch.float32: "float", torch.float64: "double", torch.float16: "half", torch.bfloat16: "__nv_bfloat16"}
    scalar = types[dtype]
    value = "(double)value" if dtype == torch.float64 else "(float)value"
    pointers = ', '.join(f'const T* x{j}' for j in range(len(sizes)))
    grad_pointers = ', '.join(f'const T* g{j}' for j in range(len(sizes)))
    reads, grad_reads, offset = [], [], 0
    for j, size in enumerate(sizes):
        prefix = 'if' if j == 0 else 'else if'
        reads.append(f'{prefix} (i < {offset + size}LL) value = x{j}[i - {offset}LL];')
        grad_reads.append(f'{prefix} (i < {offset + size}LL) gradient = g{j} ? g{j}[i - {offset}LL] : (T)0.0f;')
        offset += size
    source = r'''
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    typedef SCALAR T;
    extern "C" __global__ void repair(POINTERS, const T* fallback, T* y,
        long long* counts, long long n, long long fallback_n, long long stat_start) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        T value = (T)0.0f;
        if (i < n) { READS }
        bool bad = i < n && !isfinite(VALUE);
        if (i < n) y[i] = bad ? (i < fallback_n ? fallback[i] : (T)0.0f) : value;
        __shared__ unsigned int all_bad[256];
        __shared__ unsigned int stat_bad[256];
        all_bad[threadIdx.x] = bad;
        stat_bad[threadIdx.x] = bad && i >= stat_start;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                all_bad[threadIdx.x] += all_bad[threadIdx.x + stride];
                stat_bad[threadIdx.x] += stat_bad[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && all_bad[0]) {
            atomicAdd((unsigned long long*)counts, (unsigned long long)all_bad[0]);
            atomicAdd((unsigned long long*)(counts + 1), (unsigned long long)stat_bad[0]);
        }
    }
    extern "C" __global__ void repair_backward(POINTERS, GRAD_POINTERS, T* dx,
        T* df, long long n, long long fallback_n) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        T value = (T)0.0f;
        T gradient = (T)0.0f;
        READS
        GRAD_READS
        bool bad = !isfinite(VALUE);
        dx[i] = bad ? (T)0.0f : gradient;
        if (i < fallback_n) df[i] = bad ? gradient : (T)0.0f;
    }
    '''.replace("SCALAR", scalar).replace("VALUE", value).replace("GRAD_POINTERS", grad_pointers).replace("POINTERS", pointers).replace("GRAD_READS", '\n'.join(grad_reads)).replace("READS", '\n'.join(reads))
    return cp, cp.RawKernel(source, "repair"), cp.RawKernel(source, "repair_backward")


def _launch(kernel, tensor, sizes, args):
    cp, _, _ = _kernels(tensor.dtype, sizes)
    with cp.cuda.Device(tensor.device.index):
        torch_stream = torch.cuda.current_stream(tensor.device)
        stream = (cp.cuda.Stream.from_external(_TorchStream(torch_stream)) if hasattr(cp.cuda.Stream, "from_external")
                  else cp.cuda.ExternalStream(torch_stream.cuda_stream))
        kernel(((sum(sizes) + 255) // 256,), (256,), args, stream=stream)


class _Repair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fallback, stat_start, *values):
        values = tuple(value.contiguous() for value in values)
        sizes = tuple(value.numel() for value in values)
        first = values[0]
        _, kernel, _ = _kernels(first.dtype, sizes)
        output = first.new_empty(sum(sizes))
        counts = torch.zeros(2, device=first.device, dtype=torch.int64)
        _launch(kernel, first, sizes, (*(value.data_ptr() for value in values), fallback.data_ptr(), output.data_ptr(),
                              counts.data_ptr(), sum(sizes), fallback.numel(), stat_start))
        ctx.save_for_backward(*values)
        ctx.fallback_n = fallback.numel()
        ctx.shapes = [value.shape for value in values]
        ctx.sizes = sizes
        ctx.set_materialize_grads(False)
        outputs = [part.view_as(value) for part, value in zip(output.split(ctx.sizes), values)]
        ctx.mark_non_differentiable(counts, *(out for i, out in enumerate(outputs) if not ctx.needs_input_grad[i + 2]))
        return (*outputs, counts)

    @staticmethod
    def backward(ctx, *grads):
        values = ctx.saved_tensors
        first = values[0]
        if torch.is_grad_enabled():
            # Preserve higher-order differentiation with native operations.
            flat = torch.cat([value.reshape(-1) for value in values])
            grad = torch.cat([g.reshape(-1) if g is not None else first.new_zeros(size)
                              for g, size in zip(grads[:-1], ctx.sizes)])
            bad = ~torch.isfinite(flat)
            dx = torch.where(bad, 0.0, grad)
            df = torch.where(bad[:ctx.fallback_n], grad[:ctx.fallback_n], 0.0)
        else:
            _, _, kernel = _kernels(first.dtype, ctx.sizes)
            grads = tuple(g.contiguous() if g is not None else None for g in grads[:-1])
            dx = first.new_empty(sum(ctx.sizes))
            df = first.new_empty(ctx.fallback_n)
            _launch(kernel, first, ctx.sizes, (*(value.data_ptr() for value in values),
                                  *(g.data_ptr() if g is not None else 0 for g in grads),
                                  dx.data_ptr(), df.data_ptr(), sum(ctx.sizes), ctx.fallback_n))
        input_grads = [part.view(shape) if ctx.needs_input_grad[i + 2] else None
                       for i, (part, shape) in enumerate(zip(dx.split(ctx.sizes), ctx.shapes))]
        return df if ctx.needs_input_grad[0] else None, None, *input_grads


def repair_transition(entries, stat_start):
    """Repair transition values; only the hidden entry has a nonzero fallback.

    Returns repaired values and GPU counts [all elements, stat elements].
    Shape/dtype grouping keeps casts and autograd identical to torch.where.
    """
    groups = {}
    values = [entry[0] for entry in entries]
    for index, (tensor, fallback, allow_negative_inf) in enumerate(entries):
        if allow_negative_inf or (index > 0 and fallback is not None):
            raise ValueError("Transition CUDA repair only supports a hidden fallback and finite values")
        if not tensor.dtype.is_floating_point:
            continue
        if tensor.device.type != "cuda" or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Transition CUDA repair requires CUDA float tensors")
        if tensor.numel():
            groups.setdefault((tensor.device, tensor.dtype), []).append(index)
    counts = []
    for indices in groups.values():
        first = values[indices[0]]
        fallback = entries[0][1].to(first).reshape(-1) if indices[0] == 0 and entries[0][1] is not None else first.new_empty(0)
        if fallback.numel() and fallback.numel() != entries[0][0].numel():
            raise ValueError("Hidden fallback must have the same shape")
        stat_offset = sum(values[i].numel() for i in indices if i < stat_start)
        repaired = _Repair.apply(fallback, stat_offset, *(values[i] for i in indices))
        for i, value in zip(indices, repaired[:-1]):
            values[i] = value
        counts.append(repaired[-1])
    if not counts:
        return values, torch.zeros(2, device=entries[0][0].device, dtype=torch.int64)
    return values, counts[0] if len(counts) == 1 else torch.stack(counts).sum(0)
