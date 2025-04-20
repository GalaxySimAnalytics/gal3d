import json
import os
import logging

logger = logging.getLogger("gal3d.plugins")
PLUGINS_JSON_FILE = os.path.join(os.path.dirname(__file__), "plugins.json")

PLUGIN_CATEGORIES = ["DensityEstimator","Coordinate", "Geometry","Optimizer", "ModelProjector","Characterizer"]


def save_plugin_to_json(plugin_name: str, plugin_description: str, plugin_type: str, plugin_path: str):
    
    if plugin_type not in PLUGIN_CATEGORIES:
        logger.error(f"Invalid plugin type '{plugin_type}'. Must be one of {PLUGIN_CATEGORIES}.")
        return
    
    if os.path.exists(PLUGINS_JSON_FILE):
        with open(PLUGINS_JSON_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {category: [] for category in PLUGIN_CATEGORIES}
    
    plugin_info = {
        "name": plugin_name,
        "description": plugin_description,
        "path": plugin_path,
    }
    category_plugins = data[plugin_type]

    existing_plugins = {p["name"] for p in category_plugins}
    if plugin_name not in existing_plugins:
        category_plugins.append(plugin_info)

        with open(PLUGINS_JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Plugin '{plugin_name}' added to {plugin_type} category in {PLUGINS_JSON_FILE}")
    else:
        logger.info(f"Plugin '{plugin_name}' already exists in {plugin_type} category.")



def update_plugins_json():
    from gal3d.optimization.optimizer import Optimizer
    from gal3d.shape import Coordinate,Geometry
    from gal3d.point import DensityEstimator
    from gal3d.visualization.model_projector import ModelProjector
    from gal3d.characterization import Characterizer

    PLUGIN_BASE = [DensityEstimator,Coordinate,Geometry,Optimizer,ModelProjector,Characterizer]
    
    dic = dict(zip(PLUGIN_CATEGORIES,PLUGIN_BASE))
    for i,j in dic.items():
        all_plugins = j.available_plugins
        for name in all_plugins:
            plu = j.get_plugin(name)
            save_plugin_to_json(name,plu.__doc__,i,plu.__module__)
            

def load_plugins_info_json():
    with open(PLUGINS_JSON_FILE,'r') as f:
        plugins = json.load(f)
    return plugins
    
