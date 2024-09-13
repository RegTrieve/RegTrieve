import json
import torch
import transformers

from loguru import logger
from typing import Dict, List

from .align_utils import TOKENIZER_TO_SPECIAL_TOKEN, get_tokenizer

from transformers.generation.logits_process import LogitsProcessor

class Aligner:

    def __init__(
            self,
            base_tokenizer,
            blending_tokenizer,
            vocab_mapping
    ):
        self.base_tokenizer = base_tokenizer
        self.blending_tokenizer = blending_tokenizer
        self.vocab_mapping = vocab_mapping
        self.base_model_vocab=self.base_tokenizer.get_vocab()

    def run(self, input_ids, skip_special_tokens = True):
        aligned_ids = transform_words(
            self.base_tokenizer,
            self.blending_tokenizer,
            self.base_model_vocab,
            input_ids,
            self.vocab_mapping
        )

        # logger.debug(f"aligned_ids: {aligned_ids}")

        # decoding for str
        asr = self.base_tokenizer.batch_decode(aligned_ids, skip_special_tokens=skip_special_tokens)
        return asr

def transform_words(
    base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    base_model_vocab: Dict[str, int],
    blending_model_per_step_indices: List[List[int]],
    blending_to_base_mapping: Dict[str, str] = None,
):
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        base_model_tokenizer.__class__
    ]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_model_tokenizer.__class__
    ]

    aligned_blending_model_per_step_index = []

    # find the aligned mapping, use the corresponding logits
    # the logits and indices at this step
    # logger.debug(blending_model_per_step_indices) # batch
    for j in range(len(blending_model_per_step_indices)):
        blending_model_per_step_indices_item = blending_model_per_step_indices[j]
        aligned_blending_model_per_step_index_item = []
        for blending_index in blending_model_per_step_indices_item:
            # the token corresponds to the logit and indices
            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                [blending_index]
            )[0].replace(
                blending_model_special_token, base_model_special_token
            )

            logger.debug(f'before: {blending_t}')

            blending_t = blending_to_base_mapping[blending_t] # get mapped token in base
            logger.debug(f'after: {blending_t}')

            if blending_t in base_model_vocab:
                aligned_index = base_model_vocab[
                    blending_t
                ]  # the index of the token in base model vocab
                if (
                        aligned_index
                        #not in aligned_blending_model_per_step_index_item
                ):
                    aligned_blending_model_per_step_index_item.append(
                        aligned_index
                    )
            else:
                logger.warning(
                    f"blending_t: {blending_t} not in base_model_vocab!"
                )

        aligned_blending_model_per_step_index.append(aligned_blending_model_per_step_index_item)
    return aligned_blending_model_per_step_index


def transform_step_logits(
    base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    base_model_vocab: Dict[str, int],
    blending_model_per_step_logits: List[float],
    blending_model_per_step_indices: List[int],
    blending_to_base_mapping: Dict[str, str] = None,
):
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        base_model_tokenizer.__class__
    ]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_model_tokenizer.__class__
    ]

    aligned_blending_model_per_step_logit = []
    aligned_blending_model_per_step_index = []

    # find the aligned mapping, use the corresponding logits
    # the logits and indices at this step
    for blending_logit, blending_index in zip(
            blending_model_per_step_logits,
            blending_model_per_step_indices,
    ):
        # the token corresponds to the logit and indices
        blending_t = blending_model_tokenizer.convert_ids_to_tokens(
            [blending_index]
        )[0].replace(
            blending_model_special_token, base_model_special_token
        )
        blending_t = blending_to_base_mapping[blending_t] # get mapped token in base
        if blending_t in base_model_vocab:
            aligned_index = base_model_vocab[
                blending_t
            ]  # the index of the token in base model vocab
            if (
                    aligned_index
                    not in aligned_blending_model_per_step_index
            ):
                aligned_blending_model_per_step_index.append(
                    aligned_index
                )
                aligned_blending_model_per_step_logit.append(
                    blending_logit
                )
        else:
            logger.warning(
                f"blending_t: {blending_t} not in base_model_vocab!"
            )

    return (
        aligned_blending_model_per_step_logit,
        aligned_blending_model_per_step_index,
    )


class AlignLogitsProcessor(LogitsProcessor):

    def __init__(
            self,
            base_model_path,
            blending_model_path,
            model_max_length = 2048,
            cache_dir = "",
            blending_to_base_mapping = None
    ):
        self.base_tokenizer, _ = get_tokenizer(
            base_model_path,
            cache_dir,
            model_max_length
        )

        self.blending_tokenizer, _ = get_tokenizer(
            blending_model_path,
            cache_dir,
            model_max_length
        )

        self.blending_to_base_mapping = blending_to_base_mapping

        if self.blending_to_base_mapping is None:
            raise RuntimeError(f"Must provide the mapping: {blending_model_path}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        logger.debug(f'input_ids: {input_ids}')
        logger.debug(f'scores: {scores}')

        aligned_input_ids, aligned_scores = transform_step_logits(
            blending_model_per_step_logits=scores,
            blending_model_per_step_indices=input_ids,
            base_model_tokenizer=self.base_tokenizer,
            blending_model_tokenizer=self.blending_tokenizer,
            base_model_vocab=self.base_tokenizer.get_vocab(),
            blending_to_base_mapping=self.blending_to_base_mapping
        )

        return aligned_scores




if __name__ == "__main__":
    from align_utils import get_tokenizer

    base_model_name_or_path = ''
    blending_model_name_or_path = ''
    cache_dir = ''
    model_max_length = 2048
    vocab_mapping_save_file = 'outputs/vocab_mapping/map_vocab.json'

    # test
    base_tokenizer, _ = get_tokenizer(
        base_model_name_or_path, cache_dir, model_max_length
    )
    blending_tokenizer, _ = get_tokenizer(
        blending_model_name_or_path, cache_dir, model_max_length
    )

    with open(vocab_mapping_save_file, 'r') as f:
        blending_to_base_mapping = json.load(f)

    blending_model_per_step_logits, blending_model_per_step_indices = model() # TODO: Require model output

    transform_step_logits(
        blending_model_per_step_logits = blending_model_per_step_logits,
        blending_model_per_step_indices = blending_model_per_step_indices,
        base_model_tokenizer=base_tokenizer,
        blending_model_tokenizer=blending_tokenizer,
        base_model_vocab=base_tokenizer.get_vocab(),
        blending_to_base_mapping=blending_to_base_mapping
    )