import json
import os
import logging

logger = logging.getLogger("gal3d.plugins")
PLUGINS_JSON_FILE = os.path.join(os.path.dirname(__file__), "plugins.json")

PLUGIN_CATEGORIES = ["DensityEstimator","Coordinate", "Geometry","Optimizer", "ModelProjector","Characterizer"]


def save_plugin_to_json(plugin_name: str, plugin_description: str, plugin_type: str, plugin_path: str) -> None:
    """
    Save plugin information to a JSON file.

    This function saves the details of a plugin, including its name, description, type, 
    and path, into a JSON file. If the plugin type is invalid or required information 
    is missing, the function logs an error and exits. If the plugin already exists, 
    it will not be added again.

    Parameters
    ----------
    plugin_name : str
        The name of the plugin.
    plugin_description : str
        A brief description of the plugin.
    plugin_type : str
        The type of the plugin. Must be one of the predefined categories in `PLUGIN_CATEGORIES`.
    plugin_path : str
        The path to the plugin module.

    Returns
    -------
    None
        This function does not return any value. It writes the plugin information to a JSON file.

    Notes
    -----
    - The JSON file is located at `PLUGINS_JSON_FILE`.
    - If the JSON file does not exist, it will be created with the predefined categories.

    Examples
    --------
    >>> save_plugin_to_json(
    ...     plugin_name="ExamplePlugin",
    ...     plugin_description="An example plugin for demonstration purposes.",
    ...     plugin_type="Geometry",
    ...     plugin_path="example.plugins.geometry"
    ... )
    Plugin 'ExamplePlugin' added to Geometry category in plugins.json
    """
    if plugin_type not in PLUGIN_CATEGORIES:
        logger.error(f"Invalid plugin type '{plugin_type}'. Must be one of {PLUGIN_CATEGORIES}.")
        return
    if not plugin_name or not plugin_description or not plugin_path:
        logger.error(f"Missing plugin information.")
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
    
