from typing import Any, List, Protocol, Tuple, Union

import numpy as np
import pygame
from episode_manager.episode_manager import Action, WorldState


class VisionModule(Protocol):
    """
    Defines protocol (interface) for a vision module that is injected to
    the environment to provide vision encoding and selected action postprocessing
    """

    output_shape: Union[Tuple, List]
    high: float
    low: float

    def __call__(self, input: WorldState) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns the vision module encoded vector output based on the current step's world
        state information
        """

        return np.zeros((self.output_shape))

    def postprocess_action(self, action: Action) -> Action:
        """
        Perform any postprocessing on the action based on stored auxilliary information from the vision module
        """
        # return the same action by default

        return action

    def get_auxilliary_render(self) -> pygame.Surface:
        """
        Returns a pygame surface that visualizes auxilliary predections from the vision module
        """
        raise NotImplementedError

    def set_global_plan(self, global_plan: List[Any]):
        raise NotImplementedError
