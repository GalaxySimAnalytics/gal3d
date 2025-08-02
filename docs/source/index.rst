
欢迎来到 Gal3D 文档！ `Gal3D <https://github.com/GalaxySimAnalytics/gal3d>`_ 是专门用于给模拟中的星系提供三维形态建模的Python库。
安装Gal3D：

.. code-block:: bash

   git clone https://github.com/GalaxySimAnalytics/gal3d.git
   cd gal3d
   pip install .
如果需要以可编辑模式安装，则使用：

.. code-block:: bash

   pip install -e .

安装成功后，查阅 :ref:`教程 <tutorials>` 获取详细的使用说明。如果对 Gal3D 的开发历史感兴趣，可以查看 :ref:`开发历史 <history>`。想了解 Gal3D 各流程的具体实现原理和细节，可以查看 :ref:`实现细节 <details>`。

科学目标
========

星系的形态是一个非常重要和直观的一个星系的性质。通过对星系形态的研究，能够对星系分类，深入理解星系的形成与演化过程。传统上，
星系形态分析基于天文观测图像，星系的形态投影为二维数据（如面亮度、面密度等），并采用多种工具方法进行建模和量化。

随着N体模拟和流体力学模拟的发展，我们能够在三维空间中追踪星系的形成与演化，获得比观测更为丰富的结构和动力学信息。
然而，三维数据的复杂性也带来了新的分析挑战：如何高效、准确地量化和建模这些三维结构，成为区别于传统二维图像分析的新挑战。

**Gal3D，受观测中椭圆迭代拟合方法的启发，目标是为模拟中的星系提供高效、自动化的三维形态建模工具，帮助我们更深入理解星系在三维空间中星系结构与形态的演化过程。**


.. toctree::
   :maxdepth: 2

   教程 <tutorials/index>
   实现细节 <details/index>
   开发历史 <history/index>

   参考文档 <reference/index>
