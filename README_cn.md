### 介绍

Gal3D 是一个用于构建和分析星系三维形态模型的 Python 库，主要面向数值模拟中的粒子数据分析，可用于研究星系的三维形态结构。


### 文档

完整文档见： [readthedocs](https://gal3d.readthedocs.io)


### 安装

克隆仓库并以可编辑模式（-e）安装：

```bash
git clone https://github.com/GalaxySimAnalytics/gal3d.git
cd gal3d
pip install -e .
```

Gal3D 依赖以下库：

- **numpy**（数值计算）
- **scipy**（科学计算）
- **cython**（性能加速）
- **matplotlib**（可视化）
- **tqdm**（进度条显示）
- **h5py**（HDF5 文件支持）

可选（其他数值求解器）：

- **lmfit** (推荐)
- **nlopt**
- **optimagic**

### 使用

```python
from gal3d.analyzer import Gal3DAnalyzer

analyzer = Gal3DAnalyzer.analyze(pos,mass)

model = analyzer.fit()
```

### 开发环境设置

推荐使用 [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) 配置开发环境。


常用开发环境初始化方式如下。


**Linux / macOS**
```bash
git clone https://github.com/GalaxySimAnalytics/gal3d.git
cd gal3d
make setup
```
**Windows** (PowerShell, 无 `make`):
```powershell
git clone https://github.com/GalaxySimAnalytics/gal3d.git
cd gal3d
uv sync --extra dev --extra tests --extra optimizer
uv run pre-commit install
```

### 许可证

[MIT License](./LICENSE)