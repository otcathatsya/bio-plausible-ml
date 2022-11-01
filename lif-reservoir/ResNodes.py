from typing import Optional, Iterable, Union

import torch
from bindsnet.network.nodes import Nodes


class BasicAccumulateNode(Nodes):
    """
    Modified copy of the McCulloch-Pitts layer of neurons to represent the simplest
    acceptable type of reset behaviour to generate reservoirs with.

    :param n: The number of neurons in the layer.
    :param shape: The dimensionality of the layer.
    :param traces: Whether to record spike traces.
    :param traces_additive: Whether to record spike traces additively.
    :param tc_trace: Time constant of spike trace decay.
    :param trace_scale: Scaling factor for spike trace.
    :param sum_input: Whether to sum all inputs.
    :param reset: the voltage to reset to after a spike
    :param thresh: Spike threshold voltage.
    """

    def __init__(
            self,
            n: Optional[int] = None,
            shape: Optional[Iterable[int]] = None,
            traces: bool = False,
            traces_additive: bool = False,
            tc_trace: Union[float, torch.Tensor] = 20.0,
            trace_scale: Union[float, torch.Tensor] = 1.0,
            sum_input: bool = False,
            reset: Union[float, torch.Tensor] = -65,
            thresh: Union[float, torch.Tensor] = -52,
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.

        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v += x  # Voltages are equal to the inputs.
        self.s = self.v >= self.thresh  # Check for spiking neurons.

        self.v.masked_fill_(self.s, self.reset)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)
