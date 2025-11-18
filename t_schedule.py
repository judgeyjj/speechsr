import torch


def build_linear_quadratic_t_schedule(
    num_steps: int,
    emulate_linear_steps: int,
    sigma_max: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    构建 MovieGen / SAGA-SR 风格的 linear-quadratic t-schedule。

    参考 MovieGen 论文描述：
    - 给定一条长度为 N 的线性 schedule（从 sigma_max 到 0）；
    - 用 S 个步长近似：
      - 前 S/2 个步长：直接使用线性 schedule 的前 S/2 步；
      - 后 S/2 个步长：用二次分布在区间 [S/2, N] 上选取索引。

    Args:
        num_steps: 采样总步数 S（例如 100）
        emulate_linear_steps: 被近似的线性步数 N（例如 250）
        sigma_max: 最大 t 值（默认 1.0）
        device: 返回张量所在设备

    Returns:
        t_schedule: [S+1] 的 1D 张量，从 sigma_max 递减到 0
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    if emulate_linear_steps < num_steps:
        # 降级：如果 N 太小，就直接退化为线性 schedule
        return torch.linspace(sigma_max, 0.0, num_steps + 1, device=device)

    N = int(emulate_linear_steps)
    S = int(num_steps)

    # 原始 N 步线性 schedule（N+1 个点）
    t_linear = torch.linspace(sigma_max, 0.0, N + 1, device=device)

    # 前半部分：直接拷贝前 L 步（包含起点，因此是 L+1 个点）
    L = S // 2
    t_first = t_linear[: L + 1]  # 0..L

    # 后半部分：用 Q = S - L 个二次分布的索引覆盖 [L, N]
    Q = S - L
    if Q <= 0:
        # 极端情况：S==1 等，退化为线性
        return t_linear[: S + 1]

    # u in (0,1]，长度 Q
    u = torch.linspace(0.0, 1.0, Q + 1, device=device)[1:]
    # 二次映射到 [L, N] 区间
    idx_tail = (L + (N - L) * (u ** 2)).round().long()
    idx_tail = torch.clamp(idx_tail, L + 1, N)
    t_tail = t_linear[idx_tail]

    t_schedule = torch.cat([t_first, t_tail], dim=0)

    # 保险检查：长度必须是 S+1
    if t_schedule.numel() != S + 1:
        raise RuntimeError(
            f"Linear-quadratic schedule length mismatch: got {t_schedule.numel()}, expected {S+1}"
        )

    return t_schedule


