"""
Plugin management system for gal3d
==================================


Overview
--------
This module provides a lightweight, type-safe plugin system with three layers:

- Plugin: a concrete implementation that provides some functionality.
- PluginManager: the owner for a plugin type; it registers and exposes its plugins.
- PluginManagerRegistry: a central index of all PluginManager subclasses (one per
  plugin type) so you can discover and inspect everything in one place.


Key ideas
---------
- Managers are auto-registered with the central registry on class creation.
- Plugins are lazily discovered: a manager imports its plugin module when you
  first request a list or a specific plugin.
- You decide where your managers live (via `config.plugin_modules.modules`)
  and where each manager’s plugin implementations live (`_plugin_module`).

Examples
-----------
1) Define a plugin base and a manager for that plugin type.

>>> from gal3d.plugin import PluginBase, PluginManager
>>> class MyPlugin(PluginBase):
...     # Auto-register every subclass of MyPlugin with MyPluginManager
...     def __init_subclass__(cls, **kwargs):
...         super().__init_subclass__(**kwargs)
...         MyPluginManager.register(cls)
...
...     pass

>>> class MyPluginManager(PluginManager[MyPlugin]):
...     # Required attributes (see notes below)
...     _plugins = {}  # name -> class
...     _plugin_module = "gal3d.my_plugins"  # where implementations live
...     _base_class = MyPlugin  # enforcement for registration


2) Put actual implementations in the manager’s `_plugin_module` and let class
   creation auto-register them:


>>> # gal3d/my_plugins.py
>>> from gal3d.plugin import PluginBase  # optional if importing MyPlugin directly
>>> from gal3d.my_api import needs  # your own imports
>>> from gal3d.your_managers import MyPlugin, MyPluginManager
>>>
>>> class Foo(MyPlugin):
...     pass
>>>
>>> class Bar(MyPlugin):
...     pass

3) Make sure the manager class is importable so it gets registered in the
   central registry. A convenient way is to include the module that defines
   your manager in `config.plugin_modules.modules`:

>>> from gal3d import config
>>> config.plugin_modules.add_module("gal3d.your_managers")

4) Use the system:

>>> from gal3d.plugin import PluginManagerRegistry
>>> # Discover all managers and their plugins
>>> PluginManagerRegistry.print_plugins()

>>> # Or work with a specific manager directly
>>> MyPluginManager.available_plugins()
['Bar', 'Foo']
>>> Cls = MyPluginManager.get_plugin("Foo")
>>> plugin = Cls()  # instantiate your plugin

Terminology and configuration
-----------------------------
- Manager registration: Any subclass of `PluginManager` auto-registers itself
  with `PluginManagerRegistry` when it’s imported. Ensure the module that
  defines your manager is importable (e.g., via `config.plugin_modules.modules`).
- Plugin discovery: A manager declares `_plugin_module`, the import path that
  contains all plugin implementations for that manager. The manager imports this
  module on-demand (lazy) the first time you ask for plugins.
- Auto-registration of plugins: Typically, the plugin base (e.g., `MyPlugin`)
  overrides `__init_subclass__` to call `MyPluginManager.register(cls)`, so any
  subclass defined in `_plugin_module` is automatically registered.

Design notes
------------
- Thread-safety: Manager registration in the central registry is protected by a lock.
- Errors: lookups raise `LookupError` for unknown managers/plugins and
  `ImportError` if modules can’t be imported. Error messages are explicit.
- Logging: use logger "gal3d.plugin" for diagnostics; `print_plugins` uses a
  dedicated logger to format console output.

Troubleshooting
---------------
- “No plugin managers registered”: ensure the modules containing your
  `PluginManager` subclasses are imported (via `config.plugin_modules.modules`
  or a normal import path in your app).
- “Plugin 'X' not found in ManagerY”: verify the class name matches exactly and
  that the plugin module (`_plugin_module`) was imported successfully.
- Import errors: confirm `PYTHONPATH` and package structure align with the
  declared module paths.
"""

import logging
import threading
from typing import ClassVar, Generic, TypeVar

from gal3d.configuration import config

logger = logging.getLogger("gal3d.plugin")

_PluginType = TypeVar("_PluginType", bound="PluginBase")


class PluginManagerRegistry:
    """
    Central registry for all plugin managers.

    Responsibilities
    ----------------
    - Track all `PluginManager` subclasses by name.
    - Provide discovery and lookup.
    - Optionally import modules listed in `config.plugin_modules.modules` to
      load manager classes (so they can self-register).

    Typical usage
    -------------
    >>> from gal3d.plugin import PluginManagerRegistry
    >>> PluginManagerRegistry.print_plugins()  # pretty, human-readable summary
    >>> mgr_cls = PluginManagerRegistry.get_manager("MyPluginManager")

    Thread-safety
    -------------
    Manager registration is protected by an internal lock.
    """

    _managers: dict[str, type["PluginManager"]] = {}
    _lock = threading.Lock()

    @classmethod
    def register_manager(cls, manager: type["PluginManager"]) -> None:
        """
        Register a `PluginManager` subclass by its class name.

        Parameters
        ----------
        manager : type[PluginManager]
            The manager subclass to register. Normally called automatically
            from `PluginManager.__init_subclass__`.
        """
        with cls._lock:
            cls._managers[manager.__name__] = manager

    @classmethod
    def get_manager(cls, name: str) -> type["PluginManager"]:
        """
        Retrieve a registered `PluginManager` subclass by its class name.

        Parameters
        ----------
        name : str
            The class name of the manager to retrieve.

        Returns
        -------
        type[PluginManager]
            The manager class.

        Raises
        ------
        LookupError
            If no manager is registered under this name.
        """
        try:
            return cls._managers[name]
        except KeyError as err:
            raise LookupError(f"PluginManager '{name}' not found.") from err

    @classmethod
    def all_managers(cls) -> dict[str, type["PluginManager"]]:
        """
        Ensure manager modules are imported (if configured) and return all managers.

        It attempts to import each module path listed in
        `config.plugin_modules.modules` so that any `PluginManager` declared there
        will run its `__init_subclass__` and self-register.

        Returns
        -------
        dict[str, type[PluginManager]]
            A shallow copy of the registered managers mapping.
        """
        import importlib

        for module_path in config.plugin_modules.modules:
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                logger.error("Failed to import plugin module '%s': %s", module_path, e)
                continue
        return dict(cls._managers)

    @classmethod
    def print_plugins(cls) -> None:
        """
        Print a formatted summary of all plugin managers and their plugins.

        This prints each manager and the names of the plugins it manages in a
        readable, colorized, hierarchical format (if supported by the active
        log handler). It also logs errors encountered while retrieving plugins.
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
            n_managers = len(managers.keys())
            plugin_display.info(
                string_formatter(
                    f"\n{n_managers} Plugin Managers and their Plugins:\n",
                    fg_color="bright_blue",
                    bold=True,
                    underline=True,
                )
            )
            for manager_name, manager_class in sorted(managers.items()):
                plugin_display.info(string_formatter(f"{manager_name}:", fg_color="bright_green", bold=True))

                try:
                    plugins = manager_class.available_plugins()
                    if not plugins:
                        plugin_display.info(
                            string_formatter("  - No plugins registered", fg_color="yellow", italics=True)
                        )
                    else:
                        for plugin in sorted(plugins):
                            plugin_display.info(string_formatter(f"  - {plugin}", fg_color="bright_blue"))
                except Exception as e:
                    plugin_display.info(
                        string_formatter(f"  - Error retrieving plugins: {str(e)}", fg_color="red", italics=True)
                    )
                    main_logger.error("Error retrieving plugins from %s: %s", manager_name, str(e))

                plugin_display.info("")
        finally:
            # Clean up handler
            plugin_display.removeHandler(handler)


class PluginBase:
    """
    Abstract base class for all plugins.

    Notes for plugin type authors
    -----------------------------
    - Leave this class as-is and create your own typed base (e.g., `MyPlugin`)
      that inherits from `PluginBase`.
    - In your typed base, override `__init_subclass__` to call
      `MyPluginManager.register(cls)`. This ensures every concrete subclass of
      `MyPlugin` is auto-registered with the correct manager at class creation.
      Concrete plugin authors then just subclass `MyPlugin`, with no extra setup.

    Notes for plugin implementers
    -----------------------------
    - Usually you don’t need to override `__init_subclass__`.
    - Just subclass the typed base (e.g., `class Foo(MyPlugin): ...`) in the
      module that your manager declares in `_plugin_module`.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Called when a subclass is created.

        The default implementation simply preserves normal subclass initialization.
        Registration of concrete plugin subclasses should be handled by the
        plugin type’s base class (e.g., `MyPlugin`) and its corresponding manager.
        """
        super().__init_subclass__(**kwargs)
        # Registration is handled by the specific plugin type's Manager (via typed base classes).


class PluginManager(Generic[_PluginType]):
    """
    Generic plugin manager base class.

    Subclass this to define a plugin type manager. You must set the following
    class attributes in your subclass:

    - _plugins: dict[str, type[_PluginType]]
        The internal registry mapping plugin class names to classes.
        Initialize it as an empty dict at class definition time.
    - _plugin_module: str
        The import path of the module that contains this manager’s plugin
        implementations. The manager lazily imports this module the first time
        you call `available_plugins()` or `get_plugin()`.
    - _base_class: type[_PluginType]
        The typed plugin base class (e.g., `MyPlugin`). Used to enforce that
        only valid subclasses are registered.

    Discovery model
    ---------------
    - Manager registration: When your manager subclass is created, it is
      auto-registered with `PluginManagerRegistry`.
    - Plugin discovery: When you first access the manager (list or get), it
      imports `_plugin_module`, which should define or import all concrete plugin
      classes. Those classes should auto-register via the typed base’s
      `__init_subclass__` hook.

    Examples
    --------
    >>> class MyPlugin(PluginBase):
    ...     def __init_subclass__(cls, **kwargs):
    ...         super().__init_subclass__(**kwargs)
    ...         MyPluginManager.register(cls)
    >>> class MyPluginManager(PluginManager[MyPlugin]):
    ...     _plugins = {}
    ...     _plugin_module = "gal3d.my_plugins"
    ...     _base_class = MyPlugin
    >>> MyPluginManager.available_plugins()
    ['Bar', 'Foo']
    >>> MyPluginManager.get_plugin("Foo")
    <class 'gal3d.my_plugins.Foo'>

    Error modes
    -----------
    - Registering a class that is not a subclass of `_base_class` raises `TypeError`.
    - Requesting an unknown plugin name raises `LookupError`.
    - Import failures in `_plugin_module` raise `ImportError`.
    """

    _plugins: dict[str, type[_PluginType]]
    _plugin_module: ClassVar[str]
    _base_class: type[_PluginType]

    def __init_subclass__(cls, **kwargs):
        """
        Auto-register manager subclasses with the registry and validate config.
        """
        super().__init_subclass__(**kwargs)
        if cls is not PluginManager:
            cls._check_subclass_config()
            PluginManagerRegistry.register_manager(cls)

    @classmethod
    def _check_subclass_config(cls) -> None:
        """
        Validate that the manager subclass declares required class attributes.

        Raises
        ------
        AssertionError
            If any required attribute is missing, invalid, or if the base class
            `PluginManager` is used directly.
        """
        if cls is PluginManager:
            raise AssertionError(
                "PluginManager base class should not be used directly. Please subclass and define required attributes."
            )

        if not (hasattr(cls, "_plugins") and isinstance(cls._plugins, dict)):
            raise AssertionError(f"{cls.__name__} must define _plugins as a dict")
        if not (hasattr(cls, "_plugin_module") and isinstance(cls._plugin_module, str) and cls._plugin_module):
            raise AssertionError(f"{cls.__name__} must define _plugin_module as a non-empty str")
        if not (hasattr(cls, "_base_class") and cls._base_class is not None):
            raise AssertionError(f"{cls.__name__} must define _base_class")

    @classmethod
    def register(cls, plugin_cls: type[_PluginType]) -> None:
        """
        Register a plugin class with this manager.

        Parameters
        ----------
        plugin_cls : type[_PluginType]
            The plugin class to register. Typically invoked automatically from
            the typed base class's `__init_subclass__`.

        Raises
        ------
        TypeError
            If `plugin_cls` is not a subclass of this manager’s `_base_class`.
        """
        cls._check_subclass_config()
        if not issubclass(plugin_cls, cls._base_class):
            raise TypeError(f"{plugin_cls.__name__} must be a subclass of {cls._base_class.__name__}")
        cls._plugins[plugin_cls.__name__] = plugin_cls
        logger.debug("%s found: %s", cls.__name__, plugin_cls.__name__)

    @classmethod
    def get_plugin(cls, name: str) -> type[_PluginType]:
        """
        Retrieve a plugin class by its registered name (class name).

        Notes
        -----
        - Valid names are the class names reported by `available_plugins()`.
        - Triggers lazy loading of `_plugin_module` on first use.

        Parameters
        ----------
        name : str
            The exact class name of the desired plugin.

        Returns
        -------
        type[_PluginType]
            The plugin class.

        Raises
        ------
        LookupError
            If the plugin name is not registered.
        ImportError
            If `_plugin_module` cannot be imported during lazy loading.
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
        List all available plugin names (class names), sorted alphabetically.

        Returns
        -------
        list[str]
            The sorted list of registered plugin names.

        Raises
        ------
        ImportError
            If `_plugin_module` cannot be imported during lazy loading.
        """
        cls._check_subclass_config()
        if not cls._plugins:
            cls._load_plugins()
        return sorted(cls._plugins.keys())

    @classmethod
    def _load_plugins(cls) -> None:
        """
        Lazily import the plugin implementations module to populate `_plugins`.

        On first use, imports `cls._plugin_module`. Defining plugin classes in
        that module should trigger their registration (typically via the typed
        base’s `__init_subclass__` calling `register` on this manager).

        Raises
        ------
        ImportError
            If the plugin implementations module cannot be imported.
        """
        cls._check_subclass_config()
        import importlib

        try:
            importlib.import_module(cls._plugin_module)
            logger.debug("%s loaded plugins: %s", cls.__name__, ", ".join(cls._plugins.keys()))
        except ImportError as e:
            logger.error("Failed to import plugin module '%s': %s", cls._plugin_module, e)
            raise ImportError(f"Could not load plugins for {cls.__name__} from module '{cls._plugin_module}'") from e
