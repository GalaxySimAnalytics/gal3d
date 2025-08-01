## 介绍

Gal3D 是一个用于构建和分析三维星系形态模型的 Python 库。它专为模拟中的粒子三维建模而设计，类似于观测数据中的图像椭圆拟合。

## 安装

```bash
git clone https://github.com/GalaxySimAnalytics/gal3d.git
```

以可编辑模式安装：

```bash
cd gal3d
pip install -e .
```

Gal3D 依赖以下库：

- **numpy**（数值计算）
- **scipy**（科学计算）
- **cython**（默认用于计算运行加速）
- **matplotlib**（绘图）
- **tqdm**（进度条显示）

可选依赖：

- **numba**（JIT 加速）
- **nlopt**（数值优化算法）
- **optimagic**（数值优化算法）
- **emcee** (蒙特卡洛优化算法)

