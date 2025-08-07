import logging
from abc import ABC
from typing import Dict, Type, List, TypeVar, Generic, ClassVar

logger = logging.getLogger("gal3d.plugin")

_PluginType = TypeVar("_PluginType", bound="PluginBase")


_ALL_PLUGIN_MANAGERS: Dict[str, Type["PluginManager"]] = {}


def all_plugin_manager_types() -> List[Type["PluginManager"]]:
    """Return all registered plugin manager types."""
    return list(_ALL_PLUGIN_MANAGERS.values())

class PluginBase(ABC):
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
    _plugins: Dict[str, Type[_PluginType]]
    _plugin_module: ClassVar[str]
    _base_class: Type[_PluginType]
    
    def __init_subclass__(cls,**kwargs):
        super().__init_subclass__(**kwargs)
        if cls is not PluginManager:
            cls._check_subclass_config()
            _ALL_PLUGIN_MANAGERS[cls.__name__] = cls

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
    def register(cls, plugin_cls: Type) -> None:
        """
        Register a plugin class.

        Parameters
        ----------
        plugin_cls : Type
            The plugin class to register.
        """
        cls._check_subclass_config()
        assert issubclass(plugin_cls, cls._base_class), (
            f"{plugin_cls.__name__} must be a subclass of {cls._base_class.__name__}"
        )
        cls._plugins[plugin_cls.__name__] = plugin_cls
        logger.debug(f"{cls.__name__} found: {plugin_cls.__name__}")

    @classmethod
    def get_plugin(cls, name: str) -> Type[_PluginType]:
        """
        Retrieve a plugin class by name.
        
        Notes
        -----
        See available_plugins() for valid names.
        """
        cls._check_subclass_config()
        if not cls._plugins:
            cls._load_plugins()
        return cls._plugins[name]

    @classmethod
    def available_plugins(cls) -> List[str]:
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
        logger.info(f"{cls.__name__} loaded plugins: {', '.join(cls._plugins.keys())}")