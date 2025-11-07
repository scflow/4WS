import time
import mlx.core as mx


def _bound_action(action, lo=-1.0, hi=1.0):
    # 模拟 mppi.py 的动作约束（clip），保持张量操作
    return mx.maximum(mx.minimum(action, mx.array(hi, dtype=mx.float32)), mx.array(lo, dtype=mx.float32))


def bench_vectorized(K: int, T: int, nu: int, U: object, reps: int = 5) -> float:
    """并行/向量化版本：一次性生成形状为 (K, T, nu) 的噪声并广播加到 U 上。
    返回平均耗时（秒）。"""
    # 预热，触发 JIT/内核编译
    for _ in range(2):
        noise = mx.random.normal(shape=(K, T, nu), dtype=mx.float32)
        out = _bound_action(U + noise)
        # 强制计算完成
        _ = float(mx.sum(out))

    t0 = time.perf_counter()
    for _ in range(reps):
        noise = mx.random.normal(shape=(K, T, nu), dtype=mx.float32)
        out = _bound_action(U + noise)
        # 强制计算完成，避免懒执行影响计时
        _ = float(mx.sum(out))
    t1 = time.perf_counter()
    return (t1 - t0) / reps


def bench_loop(K: int, T: int, nu: int, U: object, reps: int = 5) -> float:
    """Python 循环版本：逐个样本生成噪声并计算，模拟非并行实现。
    返回平均耗时（秒）。"""
    # 预热
    for _ in range(2):
        acc = mx.array(0.0, dtype=mx.float32)
        for _k in range(K):
            noise_k = mx.random.normal(shape=(T, nu), dtype=mx.float32)
            out_k = _bound_action(U + noise_k)
            acc = acc + mx.sum(out_k)
        _ = float(acc)

    t0 = time.perf_counter()
    for _ in range(reps):
        acc = mx.array(0.0, dtype=mx.float32)
        for _k in range(K):
            noise_k = mx.random.normal(shape=(T, nu), dtype=mx.float32)
            out_k = _bound_action(U + noise_k)
            acc = acc + mx.sum(out_k)
        _ = float(acc)
    t1 = time.perf_counter()
    return (t1 - t0) / reps


def main():
    T, nu = 30, 4
    U = mx.zeros((T, nu), dtype=mx.float32)
    Ks = [16, 64, 256, 1024]
    reps = 5
    try:
        dev = mx.default_device()
    except Exception:
        dev = "unknown"
    print(f"Default device: {dev}")
    print(f"Benchmark with T={T}, nu={nu}, reps={reps}")
    for K in Ks:
        tv = bench_vectorized(K, T, nu, U, reps)
        tl = bench_loop(K, T, nu, U, reps)
        speedup = tl / tv if tv > 0 else float('inf')
        print(f"K={K:<4d} | vectorized: {tv:.6f}s | loop: {tl:.6f}s | speedup x{speedup:.2f}")

    # 观察大样本规模的并行缩放
    K_large = 2048
    tv_large = bench_vectorized(K_large, T, nu, U, reps)
    print(f"K={K_large:<4d} | vectorized: {tv_large:.6f}s")


if __name__ == "__main__":
    main()