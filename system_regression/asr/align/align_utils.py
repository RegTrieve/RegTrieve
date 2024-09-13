import os
import transformers
import numpy as np

from loguru import logger

TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer: "▁",
    transformers.models.whisper.tokenization_whisper.WhisperTokenizer: "Ġ"
}

def get_tokenizer(model_name_or_path, cache_dir, model_max_length):
    kwargs = {
        "use_fast": False,
        "tokenizer_trust_remote_code": False,
        "model_trust_remote_code": False,
    }
    # if "llama" in model_name_or_path.lower():
    #     kwargs["use_fast"] = False
    #     kwargs["tokenizer_trust_remote_code"] = False
    #     kwargs["model_trust_remote_code"] = False
    # elif "mpt" in model_name_or_path.lower():
    #     kwargs["use_fast"] = True
    #     kwargs["tokenizer_trust_remote_code"] = True
    #     kwargs["model_trust_remote_code"] = True
    # else:
    #     raise NotImplementedError
    logger.info(f"Loading tokenizer from {os.path.basename(model_name_or_path)}.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=kwargs["use_fast"],
        trust_remote_code=kwargs["tokenizer_trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError
    logger.info(f"    bos_token: {tokenizer.bos_token}, {tokenizer.bos_token_id} ")
    logger.info(f"    eos_token: {tokenizer.eos_token}, {tokenizer.eos_token_id} ")
    logger.info(f"    unk_token: {tokenizer.unk_token}, {tokenizer.unk_token_id} ")
    logger.info(f"    pad_token: {tokenizer.pad_token}, {tokenizer.pad_token_id} ")
    logger.info(f"    cls_token: {tokenizer.cls_token}, {tokenizer.cls_token_id} ")
    logger.info(f"    sep_token: {tokenizer.sep_token}, {tokenizer.sep_token_id} ")
    logger.info(f"    mask_token: {tokenizer.mask_token}, {tokenizer.mask_token} ")
    return tokenizer, kwargs

def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """Use dynamic time wrapping to align to tokenizers, modified from:
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
