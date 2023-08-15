#currently not used but example of how to include stopping criteria in the mlflow model. This is a callback that requires an inherited class 
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # mtp-7b is trained to add "<|endoftext|>" at the end of generations
    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
    for stop_id in stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False