import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from loguru import logger

from system_regression.asr.align.align_utils import TOKENIZER_TO_SPECIAL_TOKEN
from system_regression.asr.utils import count_seq_end, filter_eos_token

class AlignModel(nn.Module):
    """
    Map the output of the model to the base model.
    """
    def __init__(
            self,
            model,
            tokenizer,
            base_model,
            base_tokenizer,
            vocab_mapper,
            reverse_vocab_mapper, # from base to current
            topk_logits = 5,
            debug = True
    ):
        super(AlignModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # current model
        self.model = model
        self.tokenizer = tokenizer
        self.model_vocab = self.tokenizer.get_vocab()
        self.model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            self.tokenizer.__class__
        ]

        # base model
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.base_model_vocab = self.base_tokenizer.get_vocab()
        self.base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            self.base_tokenizer.__class__
        ]

        # others
        self.vocab_mapper = vocab_mapper
        self.reverse_vocab_mapper = reverse_vocab_mapper
        self.debug = debug

        self.topk_logits = topk_logits

        if self.debug:
            logger.debug(f"Model: {model.__class__}")
            logger.debug(f'Model vocab size: {self.model.config.vocab_size}')
            logger.debug(f"Model special tokens: {self.tokenizer.additional_special_tokens}")
            logger.debug(f"Base vocab size: {len(self.base_model_vocab)}")
            logger.debug(f"Base special tokens: {self.base_tokenizer.additional_special_tokens}")

        logger.info(f'Align only for top {self.topk_logits} tokens.')

    @property
    def generation_config(self):
        return self.model.generation_config

    def forward(
            self,
            input_features,
            attention_mask,
            decoder_input_ids,
            mapped_input_ids
    ):

        with torch.no_grad():
            original_outputs = self.model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask
            )

        batch_size = input_features.shape[0]
        original_logits = original_outputs.logits[:, -1, :] #original_outputs[:, -1, :] # [16, 1, 100000] -> [16, 100000]
        _, new_original_token_id = torch.max(original_logits, dim=-1)
        decoder_original_input_ids = torch.cat(
            [
                decoder_input_ids,
                torch.reshape(new_original_token_id, (batch_size, 1))
            ],
            dim=-1
        )
        original_seq_end_count = count_seq_end(decoder_original_input_ids, self.tokenizer.eos_token_id)

        assert self.topk_logits <= original_logits.size(-1), "k is greater than the size of the last dimension"

        original_top_k_logits, original_top_k_indices = torch.topk(original_logits, k=self.topk_logits, dim=-1)
        original_top_k_logits = F.softmax(original_top_k_logits, dim=-1)

        original_top_k_logits = original_top_k_logits.cpu() # [16, topk]
        original_top_k_indices = original_top_k_indices.cpu() # [16, topk]

        # mapping logits
        mapped_batch_vocab_logits = []
        for batch_id in range(batch_size):
            mapped_logits, mapped_indices = [], []
            for original_logit, original_index in zip(
                original_top_k_logits[batch_id],
                original_top_k_indices[batch_id]
            ):
                """
                TODO:
                1. Check the 'a'??
                """
                if original_index == self.tokenizer.eos_token_id:
                    # if original_index not in mapped_indices:
                    mapped_indices.append(self.base_tokenizer.eos_token_id)
                    mapped_logits.append(original_logit)
                else:
                    # map to base model
                    model_world = self.tokenizer.convert_ids_to_tokens(
                        [original_index] # todo: confirm shape
                    )[0]

                    # TODO: how to process special tokens???? now is ignore
                    if model_world in self.tokenizer.additional_special_tokens:
                        # logger.debug(f'blending_word: {model_world}')
                        continue

                    # model_world = model_world.replace(
                    #     self.model_special_token, self.base_model_special_token
                    # )

                    mapped_base_word = self.vocab_mapper[model_world]

                    if mapped_base_word in self.base_model_vocab:
                        mapped_index = self.base_model_vocab[mapped_base_word]
                        if mapped_index:
                            mapped_indices.append(mapped_index)
                            mapped_logits.append(original_logit)
                    else:
                        logger.warning(
                            f"mapped_base_word: {mapped_base_word} not in base_model_vocab!"
                        )

            mapped_vocab_logits = torch.zeros(len(self.base_model_vocab))
            if len(mapped_indices) > 0:
                mapped_indices = torch.tensor(mapped_indices)
                mapped_logits = torch.tensor(mapped_logits)
                mapped_vocab_logits[mapped_indices] = mapped_logits
            mapped_batch_vocab_logits.append(mapped_vocab_logits)

        mapped_batch_vocab_logits = torch.from_numpy(
            np.array(mapped_batch_vocab_logits)
        ).to(self.model.device)

        # obtain mapped ids
        batch_size = input_features.shape[0]
        _, new_mapped_token_id = torch.max(mapped_batch_vocab_logits, dim=-1)
        decoder_mapped_input_ids = torch.cat(
            [
                mapped_input_ids,
                torch.reshape(new_mapped_token_id, (batch_size, 1))
            ],
            dim=-1
        ).to(self.model.device)

        mapped_seq_end_count = count_seq_end(decoder_mapped_input_ids, self.base_tokenizer.eos_token_id)

        return decoder_original_input_ids, original_seq_end_count == batch_size, mapped_batch_vocab_logits, decoder_mapped_input_ids, mapped_seq_end_count == batch_size

    def reverse_map_token(self, batch_base_token_id):
        # this is a batched operation

        reverse_batch_tokens = []

        for i in range(batch_base_token_id.size(0)):
            base_token_id = batch_base_token_id[i].item()
            # map to base model
            base_world = self.base_tokenizer.convert_ids_to_tokens(
                [base_token_id]  # todo: confirm shape
            )[0]

            # TODO: how to process special tokens????
            # if blending_word in self.blending_tokenizer.additional_special_tokens:
            # logger.debug(f'blending_word: {blending_word}')
            # continue

            # base_world = base_world.replace(
            #     self.base_model_special_token, self.model_special_token
            # )

            mapped_word = self.reverse_vocab_mapper[base_world]

            if mapped_word in self.model_vocab:
                mapped_index = self.model_vocab[mapped_word]
                reverse_batch_tokens.append(mapped_index)
            else:
                logger.warning(
                    f"mapped_base_word: {mapped_word} not in base_model_vocab!"
                )
                raise RuntimeError("mapped_base_word not in model_vocab!")

        return torch.from_numpy(np.array(reverse_batch_tokens)).to(batch_base_token_id.device)

    def generate_sample_batch(
            self,
            input_features,
            attention_mask=None,
            max_length=200
    ):
        ####### initialize ########
        # greedy generation now, may use other generation methods like beam search
        batch_size = input_features.shape[0]
        # current model
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(
                self.model,
                transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextForConditionalGeneration
        ):
            # current is s2t
            decoder_input_ids = torch.tensor(
                [batch_size * [eos_token_id]]
            ).reshape(batch_size, -1).to(self.model.device)
        else:
            # current is whisper
            decoder_input_ids = torch.tensor(
                [batch_size * [self.model.generation_config.decoder_start_token_id]]
            ).reshape(batch_size, -1).to(self.model.device)

        # base model
        eos_token_id = self.base_tokenizer.eos_token_id
        if isinstance(
                self.base_model,
                transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextForConditionalGeneration
        ):
            # base model is s2t
            decoder_mapped_input_ids = torch.tensor(
                [batch_size * [eos_token_id]]
            ).reshape(batch_size, -1).to(self.model.device)
        else:
            # base model is whisper
            decoder_mapped_input_ids = torch.tensor(
                [batch_size * [self.base_model.generation_config.decoder_start_token_id]]
            ).reshape(batch_size, -1).to(self.model.device)

        ######## generate ########
        for _ in range(max_length):
            decoder_input_ids, is_terminate, mapped_vocab_logits, decoder_mapped_input_ids, is_terminate_mapped = self.forward(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                mapped_input_ids=decoder_mapped_input_ids
            )

            if is_terminate: #or is_terminate_mapped:
                # TODO: confirm this termination conditions (i.e., mapped)
                if (not is_terminate) and is_terminate_mapped:
                    logger.warning('New model terminates first..')
                break


        ######## post process ########
        if isinstance(
                self.model,
                transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration
        ):
            # model is whisper
            decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id
        decoder_input_ids = filter_eos_token(decoder_input_ids, self.tokenizer.eos_token_id)

        if isinstance(
                self.base_model,
                transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration
        ):
            # base model is whisper
            decoder_mapped_input_ids[decoder_mapped_input_ids == -100] = self.base_tokenizer.pad_token_id
        decoder_mapped_input_ids = filter_eos_token(decoder_mapped_input_ids, self.base_tokenizer.eos_token_id)

        # return: mapped ids, original ids
        return (
            self.base_tokenizer.batch_decode(decoder_mapped_input_ids, skip_special_tokens=True),
            self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
        )