"""二维各向同性弹性波有限差分模拟：剪切矩张量震源。

方程采用 Virieux 速度-应力交错网格和二阶有限差分：

    rho * dvx/dt = d(sxx)/dx + d(sxy)/dy
    rho * dvy/dt = d(sxy)/dx + d(syy)/dy
    dsxx/dt = (lambda + 2*mu) * dvx/dx + lambda * dvy/dy
    dsyy/dt = lambda * dvx/dx + (lambda + 2*mu) * dvy/dy
    dsxy/dt = mu * (dvy/dx + dvx/dy) + Mxy * f(t) * delta(x-xs,y-ys)

震源矩阵为 M = [[0, Mxy], [Mxy, 0]]，因此是纯剪切震源，不含
各向同性或 CLVD 分量。边界使用海绵阻尼，适合教学和快速试验；若用于
定量地震学研究，应替换为更严格的 PML 和更高阶差分格式。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Model:
    """二维均匀介质和网格参数。长度单位为 m，时间单位为 s。"""

    nx: int = 301
    ny: int = 201
    dx: float = 10.0
    dy: float = 10.0
    dt: float = 0.0005
    nt: int = 1200
    vp: float = 3500.0
    vs: float = 2000.0
    rho: float = 2700.0
    source_x: int | None = None
    source_y: int | None = None
    frequency: float = 12.0
    moment_xy: float = 1.0e15
    sponge_width: int = 25
    sponge_strength: float = 0.015

    def validate(self) -> None:
        if min(self.nx, self.ny, self.nt, self.sponge_width) <= 0:
            raise ValueError("nx、ny、nt 和 sponge_width 必须为正数")
        if min(self.dx, self.dy, self.dt, self.vp, self.vs, self.rho, self.frequency) <= 0:
            raise ValueError("网格、时间步长、介质参数和主频必须为正数")
        if self.vs >= self.vp:
            raise ValueError("必须满足 vs < vp")
        if self.sponge_width * 2 >= min(self.nx, self.ny):
            raise ValueError("sponge_width 太大，必须小于最小网格尺寸的一半")
        courant = self.vp * self.dt * np.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2)
        if courant >= 1.0:
            raise ValueError(f"CFL={courant:.3f} 过大，请减小 dt 或增大 dx/dy")


def ricker(t: float, frequency: float) -> float:
    """零相位 Ricker 子波。"""
    tau = np.pi * frequency * (t - 1.5 / frequency)
    return float((1.0 - 2.0 * tau**2) * np.exp(-tau**2))


def make_sponge(model: Model) -> np.ndarray:
    """返回位于单元中心的二维海绵阻尼系数。"""
    distance_x = np.minimum(
        np.arange(model.nx)[:, None], np.arange(model.nx - 1, -1, -1)[:, None]
    )
    distance_y = np.minimum(
        np.arange(model.ny)[None, :], np.arange(model.ny - 1, -1, -1)[None, :]
    )
    distance = np.minimum(distance_x, distance_y)
    taper = np.clip((model.sponge_width - distance) / model.sponge_width, 0.0, 1.0)
    return np.exp(-model.sponge_strength * taper**2)


def simulate(model: Model, snapshot_stride: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """运行模拟，返回时间轴、快照时间轴和 vx 快照。"""
    model.validate()
    if snapshot_stride <= 0:
        raise ValueError("snapshot_stride 必须为正数")

    sx = model.source_x if model.source_x is not None else model.nx // 2
    sy = model.source_y if model.source_y is not None else model.ny // 2
    if not (1 <= sx < model.nx - 1 and 1 <= sy < model.ny - 1):
        raise ValueError("震源位置必须位于网格内部")

    lam = model.rho * (model.vp**2 - 2.0 * model.vs**2)
    mu = model.rho * model.vs**2
    # 交错网格：sxx/syy 位于单元中心，vx/vy 位于单元面，sxy 位于网格顶点。
    sxx = np.zeros((model.nx, model.ny), dtype=np.float64)
    syy = np.zeros_like(sxx)
    sxy = np.zeros((model.nx + 1, model.ny + 1), dtype=np.float64)
    vx = np.zeros((model.nx + 1, model.ny), dtype=np.float64)
    vy = np.zeros((model.nx, model.ny + 1), dtype=np.float64)
    sponge = make_sponge(model)
    sponge_vx = np.pad(sponge, ((0, 1), (0, 0)), mode="edge")
    sponge_vy = np.pad(sponge, ((0, 0), (0, 1)), mode="edge")
    sponge_sxy = np.pad(sponge, ((0, 1), (0, 1)), mode="edge")

    times = np.arange(model.nt, dtype=np.float64) * model.dt
    snapshot_indices = np.arange(0, model.nt, snapshot_stride, dtype=int)
    snapshots = np.empty((snapshot_indices.size, model.ny, model.nx), dtype=np.float32)
    next_snapshot = 0

    for it, time in enumerate(times):
        dvx_dx = (vx[1:, :] - vx[:-1, :]) / model.dx
        dvy_dy = (vy[:, 1:] - vy[:, :-1]) / model.dy
        sxx += model.dt * ((lam + 2.0 * mu) * dvx_dx + lam * dvy_dy)
        syy += model.dt * (lam * dvx_dx + (lam + 2.0 * mu) * dvy_dy)

        dvy_dx = (vy[1:, 1:-1] - vy[:-1, 1:-1]) / model.dx
        dvx_dy = (vx[1:-1, 1:] - vx[1:-1, :-1]) / model.dy
        sxy[1:-1, 1:-1] += model.dt * mu * (dvy_dx + dvx_dy)
        sxy[sx, sy] += model.dt * model.moment_xy * ricker(time, model.frequency)

        dsxx_dx = (sxx[1:, :] - sxx[:-1, :]) / model.dx
        dsxy_dy = (sxy[1:-1, 1:] - sxy[1:-1, :-1]) / model.dy
        vx[1:-1, :] += model.dt / model.rho * (dsxx_dx + dsxy_dy)
        dsxy_dx = (sxy[1:, 1:-1] - sxy[:-1, 1:-1]) / model.dx
        dsyy_dy = (syy[:, 1:] - syy[:, :-1]) / model.dy
        vy[:, 1:-1] += model.dt / model.rho * (dsxy_dx + dsyy_dy)

        vx *= sponge_vx
        vy *= sponge_vy
        sxx *= sponge
        syy *= sponge
        sxy *= sponge_sxy

        if next_snapshot < snapshot_indices.size and it == snapshot_indices[next_snapshot]:
            snapshots[next_snapshot] = (0.5 * (vx[:-1, :] + vx[1:, :])).T
            next_snapshot += 1

    return times, times[snapshot_indices], snapshots


def save_result(path: Path, model: Model, times: np.ndarray, snapshot_times: np.ndarray, snapshots: np.ndarray) -> None:
    """保存 NumPy 结果，便于后处理和机器学习读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=times,
        snapshot_times=snapshot_times,
        vx_snapshots=snapshots,
        moment_tensor=np.array([[0.0, model.moment_xy], [model.moment_xy, 0.0]]),
        dx=model.dx,
        dy=model.dy,
        dt=model.dt,
    )


def save_animation(
    path: Path,
    snapshot_times: np.ndarray,
    snapshots: np.ndarray,
    fps: int = 15,
    dx: float = 1.0,
    dy: float = 1.0,
) -> None:
    """将二维网格 vx 数据绘制为平面 GIF 动图。"""
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    if snapshots.shape[0] == 0:
        raise ValueError("没有可用于生成动图的快照")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx 和 dy 必须为正数")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    display_snapshots = snapshots
    ny, nx = snapshots.shape[1:]
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    mesh = axis.pcolormesh(
        x,
        y,
        display_snapshots[0],
        cmap="seismic",
        shading="nearest",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        animated=True,
    )
    color_limit = float(np.max(np.abs(display_snapshots)))
    color_limit = max(color_limit, np.finfo(np.float32).eps)
    mesh.set_clim(-color_limit*0.2, color_limit*0.2)
    figure.colorbar(mesh, ax=axis, label="vx")
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")

    def update(frame: int) -> tuple[object]:
        mesh.set_array(display_snapshots[frame].ravel())
        axis.set_title(f"2-D shear moment source, t={snapshot_times[frame]:.3f} s")
        return (mesh,)

    animation = FuncAnimation(
        figure,
        update,
        frames=snapshots.shape[0],
        interval=1000.0 / fps,
        blit=True,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="二维有限差分剪切矩张量震源模拟")
    parser.add_argument("--output", type=Path, default=Path("fd2d_shear_result.npz"))
    parser.add_argument("--animate", type=Path, help="保存 vx 波场 GIF 动图")
    parser.add_argument("--fps", type=int, default=60, help="动图帧率，默认 60")
    parser.add_argument("--plot", action="store_true", help="显示最后一个 vx 快照")
    args = parser.parse_args()

    model = Model()
    times, snapshot_times, snapshots = simulate(model)
    save_result(args.output, model, times, snapshot_times, snapshots)
    print(f"完成: {model.nx} x {model.ny} 网格, {model.nt} 步")
    print(f"震源矩阵 M = [[0, {model.moment_xy:.3e}], [{model.moment_xy:.3e}, 0]]")
    print(f"结果已保存到: {args.output}")

    if args.animate:
        save_animation(args.animate, snapshot_times, snapshots, args.fps, model.dx, model.dy)
        print(f"动图已保存到: {args.animate}")

    if args.plot:
        import matplotlib.pyplot as plt

        display_snapshot = snapshots[-1]
        x = np.arange(model.nx) * model.dx
        y = np.arange(model.ny) * model.dy
        figure, axis = plt.subplots(figsize=(8, 5))
        mesh = axis.pcolormesh(
            x,
            y,
            display_snapshot,
            cmap="seismic",
            shading="nearest",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )
        limit = max(float(np.max(np.abs(display_snapshot))), np.finfo(np.float32).eps)
        mesh.set_clim(-limit*0.2, limit*0.2)
        figure.colorbar(mesh, ax=axis, label="vx")
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_title(f"2-D shear moment source, t={snapshot_times[-1]:.3f} s")
        plt.show()


if __name__ == "__main__":
    main()