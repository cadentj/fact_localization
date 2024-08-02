from typing import Callable, Union, Any

from nnsight.intervention import InterventionProxy
from nnsight.envoy import Envoy


class FnEnvoy(Envoy):
    def __init__(
        self,
        base: Envoy,
        fn: Callable,
        inverse: Callable = None,
    ):
        super().__init__(base._module)

        self._base = base

        self._fn = fn
        self._inverse = inverse

        self._output = None
        self._input = None

    @property
    def input(self):
        if self._input is None:
            self._input = self._base.output

        return self._input

    @input.setter
    def input(self, _: Union[InterventionProxy, Any]) -> None:
        raise NotImplementedError

    @property
    def output(self):
        # Execute the function to compute the output
        if self._output is None:
            self._output = self._fn(self._base)

        # NOTE: I think I need this for gradients?
        self._inverse(self._base, self._output)

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        # Set the new output by calling the inverse
        self._inverse(self._base, value)

        # Set the output to the new value
        self._output = value
