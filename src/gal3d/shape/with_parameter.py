import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from gal3d.optimization.parameter import Parameters

logger = logging.getLogger("gal3d.shape.with_parameter")

required_attrs = ["PN", "UB", "LB"]
required_type = [tuple, dict, dict]

class WithParameter(ABC):
    """
    Abstract base class for objects that contain parameters with bounds.

    This class defines the interface for objects that need to maintain and manipulate
    parameters with lower and upper bounds. It ensures consistent parameter handling
    across different shape and coordinate implementations.

    Subclasses must define:
    - PN: Parameter names tuple
    - UB: Upper bounds dictionary
    - LB: Lower bounds dictionary

    Attributes
    ----------
    PN : ClassVar[Tuple[str, ...]]
        Names of all parameters (class attribute)
    UB : ClassVar[Dict[str, float]]
        Upper bounds for parameters (class attribute)
    LB : ClassVar[Dict[str, float]]
        Lower bounds for parameters (class attribute)
    """

    PN: ClassVar[tuple[str, ...]]
    UB: ClassVar[dict[str, float]]
    LB: ClassVar[dict[str, float]]
    _parameter_valid: bool = False

    _derived_registry: ClassVar[dict[type["WithParameter"], dict[str, Callable[["Parameters"], float]]]] = {}


    def __init__(self, **kwargs):
        self.parameters = self.create_parameters(**kwargs)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate that concrete subclasses properly define required parameter attributes.

        This method only validates concrete classes (non-abstract classes) to ensure
        they define all required parameter attributes correctly.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent __init_subclass__.
        """
        # Call parent __init_subclass__
        super().__init_subclass__(**kwargs)

        # Mark parameters as valid
        cls._parameter_valid = True

        # Skip validation for abstract classes
        if inspect.isabstract(cls):
            return

        # For concrete implementations, verify required attributes
        for attr, typ in zip(required_attrs, required_type, strict=False):
            if not hasattr(cls, attr) or not isinstance(getattr(cls, attr), typ):
                logger.error(
                    "%s is missing required attribute '%s' or has incorrect type. Expected type %s.",
                    cls.__name__,
                    attr,
                    typ
                )
                cls._parameter_valid = False
                return

        # Verify that all parameter names in PN have corresponding entries in UB and LB
        missing_in_ub = set(cls.PN) - set(cls.UB.keys())
        missing_in_lb = set(cls.PN) - set(cls.LB.keys())
        if missing_in_lb or missing_in_ub:
            logger.error(
                "%s missing bounds for parameters: "
                "lower: %s, upper: %s",
                cls.__name__,
                missing_in_lb,
                missing_in_ub
            )
            cls._parameter_valid = False
            return

        return

    @classmethod
    @abstractmethod
    def default_parameters(cls) -> "Parameters":
        """
        Create a Parameters object with the default values and bounds.

        Returns
        -------
        Parameters
            A new Parameters object initialized with default values
            and the bounds specified by UB and LB.
        """

    @classmethod
    def create_parameters(cls, **kwargs: Any) -> "Parameters":
        """
        Initialize parameters with custom values while respecting bounds.

        Parameters
        ----------
        **kwargs
            Parameter values to override defaults.

        Returns
        -------
        Parameters
            A Parameters object with the specified values,
            constrained by the defined bounds.

        Notes
        -----
        This method is useful for creating parameters with specific initial values,
        while ensuring they stay within their defined bounds.
        """
        from gal3d.optimization.parameter import Parameters
        param = Parameters(**kwargs)
        param.add_derived(cls.derived_param_funcs())

        parameters = Parameters(**{i: param[i] for i in cls.PN})
        parameters.add_derived(cls.derived_param_funcs())
        # only set bounds for parameters with original infinite bounds
        parameters.set_lb(only_infs=True, **cls.LB)
        parameters.set_ub(only_infs=True, **cls.UB)
        return parameters

    @classmethod
    def derived_param_funcs(cls) -> dict[str, Callable]:
        """Get derived parameter functions."""
        return cls._derived_registry.get(cls, {})


    @classmethod
    def derived(cls, fn):
        """
        Register a derived parameter function.

        This method allows the registration of functions that compute derived
        parameters based on the current set of parameters. Derived functions
        are stored in a registry and can be accessed later for evaluation.

        """
        if cls not in WithParameter._derived_registry:
            WithParameter._derived_registry[cls] = {}
        WithParameter._derived_registry[cls][fn.__name__] = fn
        return fn


    @classmethod
    @abstractmethod
    def estimate_parameters(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Estimate the parameters based on the provided arguments.

        This method should be implemented by subclasses to provide specific
        parameter estimation logic.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        Any
            The estimated parameters.
        """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluate the parameterized object.

        This method must be implemented by subclasses to provide functionality
        specific to the object being parameterized.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        Any
            Subclass-specific output.
        """

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        str
            String showing the class name and parameter values.
        """

        param_repr = repr(self.parameters)

        return f"<{self.__class__.__name__}|: " + param_repr[10:] + "|>"


    @property
    def _latex_equation(self) -> str:
        return ""

    @classmethod
    def PNlatex(cls) -> dict[str, str]:
        """
        Return a mapping param_name -> LaTeX label (no surrounding $).
        Subclasses may override if they need special symbols.
        """
        import re
        mapping: dict[str, str] = {}
        for name in getattr(cls, "PN", ()):
            safe = re.sub(r"_", r"\_", name)
            mapping[name] = rf"\text{{{safe}}}"
        return mapping

    @property
    def _latex_parameters(self) -> str:
        r"""
        Build a unified LaTeX parameter list using PNlatex mapping.
        Produces: a=..., \epsilon_{ab}=..., S_a=... with global format.
        """
        if not hasattr(self, "parameters"):
            return ""
        labels = self.__class__.PNlatex()
        parts: list[str] = []
        for pname in getattr(self.__class__, "PN", ()):
            if pname not in self.parameters:
                continue
            label = labels.get(pname, pname)
            val = self.parameters[pname]
            if isinstance(val, (int, float)):
                parts.append(rf"{label}={val:.2f}")
            else:
                parts.append(rf"{label}={val}")
        if not parts:
            return ""
        return r" " + r",\ ".join(parts)

    @property
    def _latex_name(self) -> str:
        return r"$\large \mathrm{" + self._name + "}$"

    @property
    def _latex_other(self) -> str:
        return ""

    @property
    def _name(self) -> str:
        return self.__class__.__name__


    def _latex_str_(
        self,
        name: str | None = None,
        equation: str | None = None,
        params: str | None = None,
        derived: str | None = None) -> str:
        if name is None:
            name = self._latex_name
        if equation is None:
            equation = self._latex_equation
        if params is None:
            params = self._latex_parameters
        if derived is None:
            derived = self._latex_other


        def _strip_math(s: str) -> str:
            if not s:
                return s
            s = s.strip()
            if s.startswith("$") and s.endswith("$"):
                return s[1:-1]
            return s

        rows: list[tuple[str, str]] = []
        if name:
            rows.append(("\\text{Name}", _strip_math(name)))
        if equation:
            rows.append(("\\text{Func}", _strip_math(equation)))
        if params:
            rows.append(("\\text{Param ("+f"{len(self.parameters)}"+")}", _strip_math(params)))
        if derived:
            rows.append(("\\text{Derived\\ param}", _strip_math(derived)))

        if not rows:
            return self.__repr__()

        # Build aligned environment: label & : & value \\
        lines = [f"{lbl} & : & {val} \\\\" for lbl, val in rows]
        body = "\n".join(lines)
        return body

    def _repr_latex_(
        self,
        name: str | None = None,
        equation: str | None = None,
        params: str | None = None,
        derived: str | None = None) -> str:
        """
        Render an aligned LaTeX block with vertically aligned colons:
          Name  : <name>
          Func  : <equation>
          Param : <parameter values>
          Derived param : <definitions>
        Falls back to plain repr if nothing available.
        """
        body = self._latex_str_(
            name=name,
            equation=equation,
            params=params,
            derived=derived
        )
        # Use \large once at top to avoid repeating inside each fragment
        return (
            "$\n"
            "\\large\n"
            "\\begin{array}{r c l}\n" +
            body +
            "\n\\end{array}\n"
            "$"
        )


    def __getitem__(self, key: str) -> Any:
        """
        Access parameter values using dictionary-style indexing.

        Parameters
        ----------
        key : str
            Parameter name to access.

        Returns
        -------
        Any
            The value of the requested parameter.

        Raises
        ------
        KeyError
            If parameter does not exist or cannot be accessed.
        """
        if hasattr(self, "parameters") and self.parameters is not None:
            return self.parameters[key]
        raise KeyError(f"Parameter '{key}' does not exist or cannot be accessed in {self.__class__.__name__}")

    def keys(self):
        """
        Return a list of parameter names.

        Returns
        -------
        list of str
            List of keys in the Parameters object.
        """

        return list(self.parameters.keys())

    def __contains__(self, item):
        """
        Check if a parameter exists.

        Parameters
        ----------
        item : str
            The name of the parameter to check.

        Returns
        -------
        bool
            True if the parameter exists, False otherwise.
        """

        return item in self.parameters
