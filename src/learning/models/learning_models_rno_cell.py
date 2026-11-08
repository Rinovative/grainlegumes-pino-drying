"""Provide the project-owned RNO cell with a bounded tanh candidate state.

Responsibilities:
  - Define the neuraloperator RNO cell variant whose candidate state is ``tanh`` instead of ``selu``
  - Convert the cells of an already built neuraloperator RNO in place without touching parameters

Design principles:
  - Only the candidate activation deviates from the library cell; gates, parameters and state_dict keys are unchanged
  - The conversion fails closed when the library layout does not yield exactly the expected cell count
  - Real-valued data only: the project never enables ``complex_data`` for the RNO

This module does NOT:
  - Build RNO models or validate model configuration. ``learning_models_factory`` owns them
  - Change initialization, skip handling or hidden-state carry
"""

import torch
from neuralop.layers import rno_block


class TanhCandidateRNOCell(rno_block.RNOCell):
    """RNO cell whose candidate state is ``tanh`` (classic GRU candidate), not SELU."""

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return the next hidden state with a tanh candidate.

        Parameters
        ----------
        x : torch.Tensor
            Input function at the current step, shape ``(batch, hidden_channels, *spatial)``.
        h : torch.Tensor
            Previous hidden state, shape ``(batch, hidden_channels, *spatial)``.

        Returns
        -------
        torch.Tensor
            Next hidden state. Because it is a convex combination of ``h`` and a value in
            ``(-1, 1)``, ``|h_next| <= max(|h|, 1)`` holds elementwise.

        """
        update_gate = torch.sigmoid(self.input_gates[0](x) + self.hidden_gates[0](h) + self.biases[0])
        reset_gate = torch.sigmoid(self.input_gates[1](x) + self.hidden_gates[1](h) + self.biases[1])
        h_combined = self.input_gates[2](x) + self.hidden_gates[2](reset_gate * h) + self.biases[2]
        return (1.0 - update_gate) * h + update_gate * torch.tanh(h_combined)


def use_tanh_candidate_cells(model: torch.nn.Module, *, n_layers: int) -> None:
    """Swap every RNO block's cell to :class:`TanhCandidateRNOCell` in place.

    Parameters
    ----------
    model : torch.nn.Module
        neuraloperator RNO whose blocks hold library ``RNOCell`` instances.
    n_layers : int
        Number of RNO blocks the model must contain.

    Raises
    ------
    RuntimeError
        If a block holds another cell class than the library ``RNOCell``, or the number of
        converted cells differs from ``n_layers``.

    """
    converted = 0
    for module in model.modules():
        if isinstance(module, rno_block.RNOBlock):
            # Only the exact library cell has the forward this override replaces.
            if type(module.cell) is not rno_block.RNOCell:
                message = f"Expected a library RNOCell to convert, found {type(module.cell).__name__}."
                raise RuntimeError(message)
            module.cell.__class__ = TanhCandidateRNOCell
            converted += 1
    if converted != n_layers:
        message = f"Expected {n_layers} RNO cell(s) to convert to the tanh candidate, found {converted}."
        raise RuntimeError(message)
