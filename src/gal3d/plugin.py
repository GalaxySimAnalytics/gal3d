"""
Plugin management system for gal3d.

This module defines a generic plugin management system for the gal3d package.
It includes base classes for plugins and plugin managers, as well as a central
registry to track all plugin managers and their associated plugins.

Examples
--------
To create a new plugin type, define a subclass of `PluginBase` and a corresponding
`PluginManager` subclass. Register plugins by subclassing the plugin base class.

>>> from gal3d.plugin import PluginBase, PluginManager
>>> class MyPlugin(PluginBase):
...     def __init_subclass__(cls, **kwargs):
...         super().__init_subclass__(**kwargs)
...         MyPluginManager.register(cls)
...     pass

>>> class MyPluginManager(PluginManager[MyPlugin]):
...     _plugins = {}
...     _plugin_module = "gal3d.my_plugins"
...     _base_class = MyPlugin


To list all available plugins for all managers:

>>> from gal3d.plugin import PluginManagerRegistry
>>> PluginManagerRegistry.print_plugins()
"""

import logging
import threading
from typing import ClassVar, Generic, TypeVar

from gal3d.config import config

logger = logging.getLogger("gal3d.plugin")

_PluginType = TypeVar("_PluginType", bound="PluginBase")

class PluginManagerRegistry:
    """
    Central registry for all plugin managers and their modules.

    This class tracks all plugin manager types, their associated plugin modules,
    and provides unified import and lookup functionality.
    """
    _managers: dict[str, type["PluginManager"]] = {}
    _lock = threading.Lock()

    @classmethod
    def register_manager(cls, manager: type["PluginManager"]) -> None:
        with cls._lock:
            cls._managers[manager.__name__] = manager

    @classmethod
    def get_manager(cls, name: str) -> type["PluginManager"]:
        try:
            return cls._managers[name]
        except KeyError as err:
            raise LookupError(f"PluginManager '{name}' not found.") from err

    @classmethod
    def all_managers(cls) -> dict[str, type["PluginManager"]]:
        import importlib
        for module_path in config.plugin_modules.modules:
            importlib.import_module(module_path)
        return dict(cls._managers)

    @classmethod
    def print_plugins(cls) -> None:
        """
        Print a formatted summary of all plugin managers and their plugins.

        This function displays each plugin manager along with all plugins it manages
        in a readable hierarchical format, making it easy to see the plugin structure.
        Outputs are color-coded for better readability while also logging activity.
        """
        from .log import DirectOutputHandler
        from .util.string_format import string_formatter

        # Create a dedicated logger
        plugin_display = logging.getLogger("gal3d.plugin.display")
        plugin_display.setLevel(logging.INFO)
        plugin_display.propagate = False  # Prevent messages from propagating to parent logger

        # Add direct output handler
        handler = DirectOutputHandler()
        plugin_display.addHandler(handler)

        # Log to main logger
        main_logger = logging.getLogger("gal3d.plugin")

        try:
            managers = cls.all_managers()
            if not managers:
                main_logger.warning("No plugin managers registered")
                plugin_display.info(string_formatter("No plugin managers registered.", fg_color="yellow", bold=True))
                return

            plugin_display.info(string_formatter("\nPlugin Managers and their Plugins:\n",
                                            fg_color="bright_blue", bold=True, underline=True))
            for manager_name, manager_class in sorted(managers.items()):
                plugin_display.info(string_formatter(f"{manager_name}:",
                                                fg_color="bright_green", bold=True))

                try:
                    plugins = manager_class.available_plugins()
                    if not plugins:
                        plugin_display.info(string_formatter("  - No plugins registered",
                                                        fg_color="yellow", italics=True))
                    else:
                        for plugin in sorted(plugins):
                            plugin_display.info(string_formatter(f"  - {plugin}", fg_color="bright_blue"))
                except Exception as e:
                    plugin_display.info(string_formatter(f"  - Error retrieving plugins: {str(e)}",
                                                    fg_color="red", italics=True))
                    main_logger.error("Error retrieving plugins from %s: %s", manager_name, str(e))

                plugin_display.info("")
        finally:
            # Clean up handler
            plugin_display.removeHandler(handler)

class PluginBase:
    """
    Abstract base class for all plugins.

    This class should be inherited by all plugin implementations. It provides a unified
    interface and ensures that plugins can be managed by a corresponding PluginManager.

    Notes
    -----
    Subclasses should implement __init_subclass__ and call super().__init_subclass__().
    Registration is handled by the specific plugin type's Manager, not here.
    """
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Called when a subclass is created. Ensures proper initialization.
        Registration is handled by the specific plugin type's Manager.
        """
        super().__init_subclass__(**kwargs)
         # Registration is handled by the specific plugin type's Manager
        # No registration is done here

class PluginManager(Generic[_PluginType]):
    """
    Generic plugin manager base class.

    This class provides a unified interface for registering, retrieving, and listing plugins.
    Subclasses should set the following class attributes:
        - _plugins: Dict[str, Type[PluginBase]]
            A dictionary mapping plugin names to their classes.
        - _plugin_module: str
            The module path where plugin implementations are located.
        - _base_class: Type[PluginBase]
            The abstract base class for this plugin type.

    Methods
    -------
    register(plugin_cls)
        Register a plugin class.
    get_plugin(name)
        Retrieve a plugin class by name.
    available_plugins()
        List all available plugin names.
    """
    _plugins: dict[str, type[_PluginType]]
    _plugin_module: ClassVar[str]
    _base_class: type[_PluginType]

    def __init_subclass__(cls,**kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not PluginManager:
            cls._check_subclass_config()
            PluginManagerRegistry.register_manager(cls)


    @classmethod
    def _check_subclass_config(cls) -> None:
        """
        Check that the subclass has properly set required class attributes.

        Raises
        ------
        AssertionError
            If any required attribute is missing or invalid.
        """
        if cls is PluginManager:
            raise AssertionError("PluginManager base class should not be used directly. Please subclass and define required attributes.")

        if not (hasattr(cls, "_plugins") and isinstance(cls._plugins, dict)):
            raise AssertionError(f"{cls.__name__} must define _plugins as a dict")
        if not (hasattr(cls, "_plugin_module") and isinstance(cls._plugin_module, str) and cls._plugin_module):
            raise AssertionError(f"{cls.__name__} must define _plugin_module as a non-empty str")
        if not (hasattr(cls, "_base_class") and cls._base_class is not None):
            raise AssertionError(f"{cls.__name__} must define _base_class")

    @classmethod
    def register(cls, plugin_cls: type[_PluginType]) -> None:
        """
        Register a plugin class.

        Parameters
        ----------
        plugin_cls : Type[_PluginType]
            The plugin class to register.
        """
        cls._check_subclass_config()
        if not issubclass(plugin_cls, cls._base_class):
            raise TypeError(f"{plugin_cls.__name__} must be a subclass of {cls._base_class.__name__}")
        cls._plugins[plugin_cls.__name__] = plugin_cls
        logger.debug("%s found: %s", cls.__name__, plugin_cls.__name__)

    @classmethod
    def get_plugin(cls, name: str) -> type[_PluginType]:
        """
        Retrieve a plugin class by name.

        Notes
        -----
        See available_plugins() for valid names.
        """
        cls._check_subclass_config()
        if not cls._plugins:
            cls._load_plugins()
        try:
            return cls._plugins[name]
        except KeyError as err:
            raise LookupError(f"Plugin '{name}' not found in {cls.__name__}") from err

    @classmethod
    def available_plugins(cls) -> list[str]:
        """
        List all available plugin names.

        Returns
        -------
        List[str]
            A list of all registered plugin class names.
        """
        cls._check_subclass_config()
        if not cls._plugins:
            cls._load_plugins()
        return list(cls._plugins.keys())

    @classmethod
    def _load_plugins(cls):
        """
        Dynamically import the plugin module to register all available plugins.

        This method imports the module specified by _plugin_module, which should
        trigger registration of all plugin classes in that module.

        Raises
        ------
        ImportError
            If the plugin module cannot be imported.
        """
        cls._check_subclass_config()
        import importlib
        importlib.import_module(cls._plugin_module)
        logger.debug("%s loaded plugins: %s", cls.__name__, ", ".join(cls._plugins.keys()))
