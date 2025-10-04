## 介绍

Gal3D 用于构建和分析星系或暗晕的三维形态模型的 Python 库。它专为模拟中的粒子三维形态分析而设计，类似于天文观测数据中的图像椭圆拟合方法。

## 安装

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

## 使用

```python
from gal3d.analyzer import Gal3DAnalyzer

analyzer = Gal3DAnalyzer.analyze(pos,mass)

model = analyzer.fit()
```

使用示例见 [gal3d_example](https://github.com/GalaxySimAnalytics/gal3d_example)，或参考文档获取详细说明。

## 许可证

[MIT License](./LICENSE)