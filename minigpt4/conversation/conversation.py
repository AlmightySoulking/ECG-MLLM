"""
Conversation module for MiniGPT models.
"""

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):
    """
    Custom stopping criteria that stops generation when stop words are encountered.
    """

    def __init__(self, stops, encounters=1):
        """
        Initialize the stopping criteria.
        
        Args:
            stops: List of tensors containing stop word token ids
            encounters: Number of encounters before stopping (default: 1)
        """
        super().__init__()
        self.stops = stops
        self.encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """
        Check if stopping criteria is met.
        
        Args:
            input_ids: Generated token ids
            scores: Logits/scores for next token
            
        Returns:
            Boolean indicating whether to stop generation
        """
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
