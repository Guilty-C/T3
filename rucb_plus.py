# rucb_plus.py
# Robust Sliding-Window Bernstein-UCB (RUCB+)
from dataclasses import dataclass
import numpy as np
from collections import deque

def median_of_means(x: np.ndarray, b: int = 8):
    """Robust mean/var via Median-of-Means (MOM)."""
    n = len(x)
    if n == 0:
        return 0.0, 1.0
    b = max(2, min(b, n))
    m = n // b
    if m == 0:
        return float(np.mean(x)), float(np.var(x) + 1e-8)
    blocks = [x[i*m:(i+1)*m] for i in range(b-1)]
    blocks.append(x[(b-1)*m:])
    means = np.array([blk.mean() for blk in blocks])
    vars_ = np.array([blk.var() if len(blk)>1 else 0.0 for blk in blocks])
    mu = float(np.median(means))
    # 用块方差的中位数作为稳健方差估计
    v = float(np.median(vars_) + 1e-8)
    return mu, v

@dataclass
class RUCBPlusConfig:
    K: int
    window_init: int = 512     # 初始滑窗
    mom_blocks: int = 8
    c_eps: float = 1.5         # 相对淘汰阈值系数
    ph_lambda_scale: float = 3.0  # Page-Hinkley 阈值系数
    ph_alpha: float = 0.01        # PH 漂移系数
    adapt_w: bool = True
    w_min: int = 128
    w_max: int = 4096
    use_catoni: bool = False     # 如需切 Catoni，可扩展此开关
    
    # Task3 specific params (Added for compatibility with run_experiments.STRICT2.py)
    min_pulls_per_arm: int = 10
    n_min_elim: int = 30
    elim_margin: float = 0.05
    recheck_period: int = 500
    enable_ph: bool = True

class PageHinkley:
    def __init__(self, delta: float, alpha: float):
        self.delta = delta
        self.alpha = alpha
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, x):
        # 经典 PH：对均值漂移敏感
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.cum = self.cum + x - self.mean - self.delta
        if self.cum < self.min_cum:
            self.min_cum = self.cum
        return (self.cum - self.min_cum) > self.delta

class RUCBPlus:
    def __init__(self, cfg: RUCBPlusConfig):
        self.cfg = cfg
        self.t = 0
        self.buffers = [deque(maxlen=cfg.window_init) for _ in range(cfg.K)]
        self.n = np.zeros(cfg.K, dtype=int)
        self.mu = np.zeros(cfg.K, dtype=float)
        self.v = np.ones(cfg.K, dtype=float)
        self.active = np.ones(cfg.K, dtype=bool)
        self.ph = [PageHinkley(delta=1.0, alpha=cfg.ph_alpha) for _ in range(cfg.K)]  # delta 会随方差自适应
        self.window = cfg.window_init
        
        # Compatibility logs for Task3
        self.step_log = []
        self.elim_log = []

    def _adapt_window(self):
        if not self.cfg.adapt_w:
            return
        # 简单自适应：随 sqrt(t) 增长，夹在 [w_min, w_max]
        target = int(np.clip(np.sqrt(max(self.t,1))*10, self.cfg.w_min, self.cfg.w_max))
        if target != self.window:
            self.window = target
            for k in range(self.cfg.K):
                # 变更 buffer 容量：重建 deque
                old = list(self.buffers[k])[-self.window:]
                self.buffers[k] = deque(old, maxlen=self.window)

    def _robust_stats(self, k):
        arr = np.array(self.buffers[k], dtype=float)
        mu, v = median_of_means(arr, b=self.cfg.mom_blocks)
        return mu, v

    def select_arm(self):
        # 计算稳健均值与 Bernstein 半径
        UCB = np.zeros(self.cfg.K)
        LCB = np.zeros(self.cfg.K)
        for k in range(self.cfg.K):
            if self.n[k] == 0:
                # 强制初始化，每臂先试一次
                UCB[k] = 1e9
                LCB[k] = -1e9
                continue
            mu, v = self._robust_stats(k)
            self.mu[k], self.v[k] = mu, v
            rad = np.sqrt(2.0 * v * np.log(max(self.t,2)) / self.n[k]) + 3.0*np.log(max(self.t,2))/self.n[k]
            UCB[k] = mu + rad
            LCB[k] = mu - rad

        # 相对淘汰
        bestLCB = np.max(LCB)
        eps_t = self.cfg.c_eps * np.sqrt(np.log(max(self.t,2)) / max(self.t,1))
        self.active = UCB >= (bestLCB - eps_t)

        # 选臂：active 中 UCB 最大
        idx = np.where(self.active)[0]
        if len(idx) == 0:
            idx = np.arange(self.cfg.K)
        k_star = idx[np.argmax(UCB[idx])]
        return int(k_star), UCB, LCB

    def update(self, k, reward):
        self.t += 1
        self._adapt_window()
        self.buffers[k].append(reward)
        self.n[k] += 1
        # 根据当前稳健方差调节 PH 阈值
        mu, v = self._robust_stats(k)
        self.mu[k], self.v[k] = mu, v
        lam = self.cfg.ph_lambda_scale * np.sqrt(max(v,1e-8))
        # 更新 PH；若触发，做局部重置（保留最近 10 个样本作为暖启动）
        triggered = PageHinkley(delta=lam, alpha=self.cfg.ph_alpha).update(reward)
        # 用自己的 PH 累积（更稳）
        trig2 = self.ph[k].update(reward)
        if triggered or trig2:
            recent = list(self.buffers[k])[-10:]
            self.buffers[k].clear()
            for x in recent:
                self.buffers[k].append(x)
            self.n[k] = len(self.buffers[k])
