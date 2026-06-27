import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 1. 网格
# =========================
n = 400
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# =========================
# 2. 初始化背景
# =========================
# 背景值稍高一点，避免底色过空
Z = np.ones_like(X) * 0.20

# =========================
# 3. 加一个大尺度平滑背景场
# =========================
# 让整幅图有整体热度起伏，更自然
Z += 0.10 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * 0.38**2))

# =========================
# 4. 构造均匀分布的热点
# =========================
# 用规则分布 + 轻微扰动，让分布看起来均匀但不死板
xs = [0.22, 0.50, 0.78]
ys = [0.25, 0.50, 0.75]

for i, cx in enumerate(xs):
    for j, cy in enumerate(ys):
        # 轻微变化，避免完全机械对称
        amp = 0.10 + 0.015 * np.sin((i + 1) * (j + 2))
        sigma = 0.085 + 0.01 * np.cos(i + j)

        Z += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

# =========================
# 5. 再补一些更宽、更弱的扩散
# =========================
# 让热点之间更容易自然连接
soft_spots = [
    (0.35, 0.65, 0.14, 0.05),
    (0.65, 0.65, 0.14, 0.05),
    (0.35, 0.35, 0.14, 0.05),
    (0.65, 0.35, 0.14, 0.05),
    (0.50, 0.50, 0.16, 0.06),
]

for cx, cy, sigma, amp in soft_spots:
    Z += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

# =========================
# 6. 归一化
# =========================
Z = (Z - Z.min()) / (Z.max() - Z.min())

# =========================
# 7. 压缩对比度
# =========================
# 减少局部过亮，让图看起来更柔和
Z = 0.12 + 0.70 * (Z ** 1.25)

# 再归一化到 0~1，给 colorbar 用
Z = (Z - Z.min()) / (Z.max() - Z.min())

# =========================
# 8. 自定义颜色
# =========================
cmap = LinearSegmentedColormap.from_list(
    "soft_heat",
    ["#dce9df", "#bfdcc9", "#e8c89d", "#f3a36f", "#ef5a45"]
)

# =========================
# 9. 绘图
# =========================
fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")

im = ax.imshow(
    Z,
    cmap=cmap,
    vmin=0,
    vmax=1,
    origin="lower",
    extent=[0, 1, 0, 1],
    interpolation="bicubic"
)

# 左边保持方形
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

# 边框样式
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.0)
    spine.set_edgecolor("#6aa894")

# 右边矩形色条
cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.07)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["0", "1"])
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(labelsize=16, width=1.2)

plt.tight_layout()
plt.show()