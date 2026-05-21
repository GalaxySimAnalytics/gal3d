
欢迎查阅 Gal3D 文档！ `Gal3D <https://github.com/GalaxySimAnalytics/gal3d>`_ 是一个面向模拟星系的三维形态建模与分析的Python库。
旨在从粒子数据中衡量质量分布的三维形态特征，帮助深入理解星系结构与形态在三维空间中的演化过程。



安装
====

克隆仓库并安装：

.. code-block:: bash

   git clone https://github.com/GalaxySimAnalytics/gal3d.git
   cd gal3d
   pip install .

如果需要以可编辑模式安装，则使用：

.. code-block:: bash

   pip install -e .



科学目标
========

星系的内禀三维形态是理解其形成历史、动力学状态与结构演化的重要线索。
从核球，到盘，到恒星晕，不同结构主导了不同半径处的质量分布，星系的自身形态也会随半径显著变化。
因此，刻画三维形态随半径的变化，是连接星系形态与物理形成过程的关键。

现代数值模拟能够再现星系形态和结构的复杂性，并提供了直接访问三维质量分布的机会。
但与此同时，模拟粒子数据也提出了新的方法学挑战：如何从含噪声、含多结构成分的三维粒子分布中，稳定地量化其径向形态的变化。

**Gal3D 正是在这一背景下开发的。它面向模拟中的粒子数据，结合平滑密度场重建、径向采样以及超椭球拟合，对星系的三维等密度结构进行建模与量化。
Gal3D 不仅能够测量轴比、方向和中心偏移，还能够描述盒状度、盘状度等高阶形态特征，从而为研究星系三维结构的径向变化、比较不同数值模拟中的形态差异，以及连接内禀三维结构与可观测投影形态，提供统一而灵活的分析框架。**



引用
====
如果你在研究或项目中使用了 Gal3D，请引用以下文献：
正式文献信息将在相关文章发表后补充更新。

::

    @software{gal3d_2026,
      author = {Shuai Lu and Min Du},
      title = {Gal3D},
      year = {2026},
      url = {https://github.com/GalaxySimAnalytics/gal3d}
    }



文档内容
========

本手册主要分为四个部分：

- :ref:`教程 <tutorials>`：介绍 Gal3D 的基本用法、主要功能及进阶示例。
- :ref:`实现细节 <details>`：详细说明各模块的实现原理与技术细节。
- :ref:`参考文档 <reference>`：提供完整的 API 参考。

.. toctree::
   :maxdepth: 2

   教程 <tutorials/index>
   实现细节 <details/index>
   参考文档 <reference/index>
