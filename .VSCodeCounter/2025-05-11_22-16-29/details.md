# Details

Date : 2025-05-11 22:16:29

Directory /home/yxi/Simulation/TNG50_try/gal3d_develop/gal3d/src

Total : 75 files,  11234 codes, 169 comments, 2411 blanks, all 13814 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [src/gal3d/\_\_init\_\_.py](/src/gal3d/__init__.py) | Python | 5 | 2 | 5 | 12 |
| [src/gal3d/\_info.py](/src/gal3d/_info.py) | Python | 29 | 0 | 2 | 31 |
| [src/gal3d/\_plugins.py](/src/gal3d/_plugins.py) | Python | 83 | 0 | 22 | 105 |
| [src/gal3d/analyzer.py](/src/gal3d/analyzer.py) | Python | 398 | 13 | 114 | 525 |
| [src/gal3d/characterization/\_\_init\_\_.py](/src/gal3d/characterization/__init__.py) | Python | 1 | 0 | 0 | 1 |
| [src/gal3d/characterization/characterizer.py](/src/gal3d/characterization/characterizer.py) | Python | 63 | 0 | 24 | 87 |
| [src/gal3d/characterization/characterizer.pyi](/src/gal3d/characterization/characterizer.pyi) | Python | 35 | 0 | 12 | 47 |
| [src/gal3d/characterization/characterizer\_plugins/\_\_init\_\_.py](/src/gal3d/characterization/characterizer_plugins/__init__.py) | Python | 1 | 0 | 0 | 1 |
| [src/gal3d/characterization/characterizer\_plugins/galaxy\_bar.py](/src/gal3d/characterization/characterizer_plugins/galaxy_bar.py) | Python | 311 | 0 | 46 | 357 |
| [src/gal3d/configuration.py](/src/gal3d/configuration.py) | Python | 166 | 1 | 29 | 196 |
| [src/gal3d/default\_config.ini](/src/gal3d/default_config.ini) | Ini | 12 | 0 | 18 | 30 |
| [src/gal3d/executor.py](/src/gal3d/executor.py) | Python | 19 | 1 | 7 | 27 |
| [src/gal3d/field/\_\_init\_\_.py](/src/gal3d/field/__init__.py) | Python | 1 | 0 | 0 | 1 |
| [src/gal3d/field/grid/\_\_init\_\_.py](/src/gal3d/field/grid/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [src/gal3d/field/grid/cartesian\_grid.py](/src/gal3d/field/grid/cartesian_grid.py) | Python | 166 | 0 | 27 | 193 |
| [src/gal3d/field/grid/util.py](/src/gal3d/field/grid/util.py) | Python | 252 | 8 | 51 | 311 |
| [src/gal3d/field/spherical\_field/\_\_init\_\_.py](/src/gal3d/field/spherical_field/__init__.py) | Python | 3 | 0 | 0 | 3 |
| [src/gal3d/field/spherical\_field/field.py](/src/gal3d/field/spherical_field/field.py) | Python | 577 | 1 | 134 | 712 |
| [src/gal3d/field/spherical\_field/ray/\_\_init\_\_.py](/src/gal3d/field/spherical_field/ray/__init__.py) | Python | 128 | 1 | 29 | 158 |
| [src/gal3d/field/spherical\_field/ray/monotonic\_profile.py](/src/gal3d/field/spherical_field/ray/monotonic_profile.py) | Python | 242 | 0 | 45 | 287 |
| [src/gal3d/field/spherical\_field/ray/util.py](/src/gal3d/field/spherical_field/ray/util.py) | Python | 279 | 4 | 53 | 336 |
| [src/gal3d/field/spherical\_field/spherical\_harmonic.py](/src/gal3d/field/spherical_field/spherical_harmonic.py) | Python | 69 | 0 | 13 | 82 |
| [src/gal3d/field/spherical\_field/spherical\_vector.py](/src/gal3d/field/spherical_field/spherical_vector.py) | Python | 119 | 5 | 37 | 161 |
| [src/gal3d/field/spherical\_field/util.py](/src/gal3d/field/spherical_field/util.py) | Python | 95 | 0 | 22 | 117 |
| [src/gal3d/optimization/\_\_init\_\_.py](/src/gal3d/optimization/__init__.py) | Python | 2 | 0 | 2 | 4 |
| [src/gal3d/optimization/optimizer.py](/src/gal3d/optimization/optimizer.py) | Python | 172 | 0 | 40 | 212 |
| [src/gal3d/optimization/optimizer.pyi](/src/gal3d/optimization/optimizer.pyi) | Python | 45 | 0 | 16 | 61 |
| [src/gal3d/optimization/optimizer\_plugins/\_\_init\_\_.py](/src/gal3d/optimization/optimizer_plugins/__init__.py) | Python | 3 | 0 | 1 | 4 |
| [src/gal3d/optimization/optimizer\_plugins/optimize\_nlopt.py](/src/gal3d/optimization/optimizer_plugins/optimize_nlopt.py) | Python | 37 | 1 | 12 | 50 |
| [src/gal3d/optimization/optimizer\_plugins/optimize\_optimagic.py](/src/gal3d/optimization/optimizer_plugins/optimize_optimagic.py) | Python | 36 | 1 | 10 | 47 |
| [src/gal3d/optimization/optimizer\_plugins/optimize\_scipy.py](/src/gal3d/optimization/optimizer_plugins/optimize_scipy.py) | Python | 51 | 1 | 9 | 61 |
| [src/gal3d/optimization/parameter.py](/src/gal3d/optimization/parameter.py) | Python | 656 | 6 | 118 | 780 |
| [src/gal3d/optimization/result.py](/src/gal3d/optimization/result.py) | Python | 385 | 2 | 80 | 467 |
| [src/gal3d/optimization/util.py](/src/gal3d/optimization/util.py) | Python | 52 | 0 | 17 | 69 |
| [src/gal3d/plugins.json](/src/gal3d/plugins.json) | JSON | 64 | 0 | 0 | 64 |
| [src/gal3d/point/\_\_init\_\_.py](/src/gal3d/point/__init__.py) | Python | 99 | 1 | 18 | 118 |
| [src/gal3d/point/density\_estimator.py](/src/gal3d/point/density_estimator.py) | Python | 197 | 1 | 43 | 241 |
| [src/gal3d/point/density\_estimator.pyi](/src/gal3d/point/density_estimator.pyi) | Python | 72 | 0 | 20 | 92 |
| [src/gal3d/point/density\_estimator\_plugins/\_\_init\_\_.py](/src/gal3d/point/density_estimator_plugins/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [src/gal3d/point/density\_estimator\_plugins/estimator\_knn.py](/src/gal3d/point/density_estimator_plugins/estimator_knn.py) | Python | 175 | 0 | 47 | 222 |
| [src/gal3d/point/global\_calculator.py](/src/gal3d/point/global_calculator.py) | Python | 193 | 0 | 44 | 237 |
| [src/gal3d/point/util.py](/src/gal3d/point/util.py) | Python | 211 | 3 | 38 | 252 |
| [src/gal3d/run\_config.ini](/src/gal3d/run_config.ini) | Ini | 41 | 0 | 56 | 97 |
| [src/gal3d/shape/\_\_init\_\_.py](/src/gal3d/shape/__init__.py) | Python | 590 | 4 | 152 | 746 |
| [src/gal3d/shape/coordinate.py](/src/gal3d/shape/coordinate.py) | Python | 166 | 0 | 44 | 210 |
| [src/gal3d/shape/coordinate.pyi](/src/gal3d/shape/coordinate.pyi) | Python | 49 | 0 | 16 | 65 |
| [src/gal3d/shape/coordinate\_plugins/\_\_init\_\_.py](/src/gal3d/shape/coordinate_plugins/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [src/gal3d/shape/coordinate\_plugins/\_rotation\_eular\_util.py](/src/gal3d/shape/coordinate_plugins/_rotation_eular_util.py) | Python | 84 | 0 | 20 | 104 |
| [src/gal3d/shape/coordinate\_plugins/euler\_shift.py](/src/gal3d/shape/coordinate_plugins/euler_shift.py) | Python | 220 | 0 | 44 | 264 |
| [src/gal3d/shape/fns.py](/src/gal3d/shape/fns.py) | Python | 164 | 6 | 26 | 196 |
| [src/gal3d/shape/geometry.py](/src/gal3d/shape/geometry.py) | Python | 239 | 0 | 57 | 296 |
| [src/gal3d/shape/geometry.pyi](/src/gal3d/shape/geometry.pyi) | Python | 109 | 0 | 30 | 139 |
| [src/gal3d/shape/geometry\_plugins/\_\_init\_\_.py](/src/gal3d/shape/geometry_plugins/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [src/gal3d/shape/geometry\_plugins/\_ellipsoid\_s\_util.py](/src/gal3d/shape/geometry_plugins/_ellipsoid_s_util.py) | Python | 351 | 4 | 44 | 399 |
| [src/gal3d/shape/geometry\_plugins/\_ellipsoid\_util.py](/src/gal3d/shape/geometry_plugins/_ellipsoid_util.py) | Python | 485 | 6 | 59 | 550 |
| [src/gal3d/shape/geometry\_plugins/\_ellipsoid\_util\_cy.pyx](/src/gal3d/shape/geometry_plugins/_ellipsoid_util_cy.pyx) | Cython | 156 | 3 | 40 | 199 |
| [src/gal3d/shape/geometry\_plugins/ellipsoid.py](/src/gal3d/shape/geometry_plugins/ellipsoid.py) | Python | 254 | 0 | 45 | 299 |
| [src/gal3d/shape/geometry\_plugins/ellipsoid\_s.py](/src/gal3d/shape/geometry_plugins/ellipsoid_s.py) | Python | 215 | 0 | 37 | 252 |
| [src/gal3d/shape/minimize\_func.py](/src/gal3d/shape/minimize_func.py) | Python | 67 | 2 | 21 | 90 |
| [src/gal3d/shape/with\_parameter.py](/src/gal3d/shape/with_parameter.py) | Python | 141 | 6 | 35 | 182 |
| [src/gal3d/util/\_\_init\_\_.py](/src/gal3d/util/__init__.py) | Python | 1 | 0 | 1 | 2 |
| [src/gal3d/util/array\_operate.py](/src/gal3d/util/array_operate.py) | Python | 165 | 21 | 30 | 216 |
| [src/gal3d/util/error\_handling.py](/src/gal3d/util/error_handling.py) | Python | 119 | 9 | 25 | 153 |
| [src/gal3d/util/func\_cache.py](/src/gal3d/util/func_cache.py) | Python | 17 | 1 | 9 | 27 |
| [src/gal3d/util/func\_decorator.py](/src/gal3d/util/func_decorator.py) | Python | 173 | 6 | 49 | 228 |
| [src/gal3d/util/func\_signature.py](/src/gal3d/util/func_signature.py) | Python | 461 | 42 | 91 | 594 |
| [src/gal3d/util/string\_format.py](/src/gal3d/util/string_format.py) | Python | 284 | 0 | 60 | 344 |
| [src/gal3d/visualization/\_\_init\_\_.py](/src/gal3d/visualization/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [src/gal3d/visualization/data\_model\_residual.py](/src/gal3d/visualization/data_model_residual.py) | Python | 456 | 2 | 46 | 504 |
| [src/gal3d/visualization/hist2d.py](/src/gal3d/visualization/hist2d.py) | Python | 268 | 4 | 52 | 324 |
| [src/gal3d/visualization/model\_projector.py](/src/gal3d/visualization/model_projector.py) | Python | 276 | 1 | 50 | 327 |
| [src/gal3d/visualization/model\_projector.pyi](/src/gal3d/visualization/model_projector.pyi) | Python | 42 | 0 | 17 | 59 |
| [src/gal3d/visualization/model\_projector\_plugins/\_\_init\_\_.py](/src/gal3d/visualization/model_projector_plugins/__init__.py) | Python | 2 | 0 | 1 | 3 |
| [src/gal3d/visualization/model\_projector\_plugins/projector\_line\_integration.py](/src/gal3d/visualization/model_projector_plugins/projector_line_integration.py) | Python | 72 | 0 | 24 | 96 |
| [src/gal3d/visualization/model\_projector\_plugins/projector\_sph\_grid.py](/src/gal3d/visualization/model_projector_plugins/projector_sph_grid.py) | Python | 59 | 0 | 20 | 79 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)