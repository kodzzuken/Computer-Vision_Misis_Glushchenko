import functools

import torch
import triton

from kernel import layernorm_forward, layernorm_forward_torch


_PROVIDER_TO_FN = {
    "triton": layernorm_forward,
    "torch-eager": layernorm_forward_torch,
}


def _make_inputs(m: int, n: int, dtype: torch.dtype):
    device = "cuda"
    x = torch.randn((m, n), device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn((n,), device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    return x, weight, bias, grad


def _zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def run_forward_benchmark() -> None:
    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=list(_PROVIDER_TO_FN.keys()),
            line_names=list(_PROVIDER_TO_FN.keys()),
            styles=[("blue", "-"), ("red", "--")],
            ylabel="GB/s примерно",
            plot_name="layernorm_forward_fp32",
            args={"M": 4096},
        )
    ])
    def benchmark(M: int, N: int, provider: str):
        dtype = torch.float32
        x, weight, bias, _ = _make_inputs(M, N, dtype)

        fn = functools.partial(_PROVIDER_TO_FN[provider], x, weight, bias)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        def gbps(t_ms: float) -> float:
            # считаем не идеально, но для сравнения норм, читаем x/weight/bias и пишем output
            total_bytes = M * N * 4 * 4 + M * 2 * 4
            return (total_bytes * 1e-9) / (t_ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def run_forward_backward_benchmark() -> None:
    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=list(_PROVIDER_TO_FN.keys()),
            line_names=list(_PROVIDER_TO_FN.keys()),
            styles=[("blue", "-"), ("red", "--")],
            ylabel="GB/s примерно",
            plot_name="layernorm_forward_backward_fp32",
            args={"M": 4096},
        )
    ])
    def benchmark(M: int, N: int, provider: str):
        dtype = torch.float32
        x, weight, bias, grad = _make_inputs(M, N, dtype)
        provider_fn = _PROVIDER_TO_FN[provider]

        def fn():
            _zero_grads(x, weight, bias)
            y = provider_fn(x, weight, bias)
            y.backward(grad)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        def gbps(t_ms: float) -> float:
            # тут тоже примерная оценка, потому что backward сложнее и есть atomic_add
            total_bytes = M * N * 4 * 9 + N * 2 * 4 + M * 2 * 4
            return (total_bytes * 1e-9) / (t_ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


if __name__ == "__main__":
    run_forward_benchmark()
    run_forward_backward_benchmark()
