"""Mapping vocabs from different models."""
import os
import json
import tqdm
import argparse
import editdistance
import multiprocessing

import numpy as np

from loguru import logger
from datasets import load_from_disk
from align_utils import get_tokenizer, TOKENIZER_TO_SPECIAL_TOKEN

def find_best_mapping(
    x,
    base_tokens, # base tokens, map new -> base
    blending_model_special_token=None,
    base_model_special_token=None,
    best_one=True,
):
    tmp_x = x.replace(blending_model_special_token, base_model_special_token)

    if tmp_x in base_tokens:
        return x, tmp_x
    else:
        if best_one:
            return x, min(
                [(y, editdistance.eval(tmp_x.lower(), y.lower())) for y in base_tokens],
                key=lambda d: d[1],
            )[0]
        else:
            token_and_distance = [
                (y, editdistance.eval(tmp_x.lower(), y.lower())) for y in base_tokens
            ]
            min_distance = min(item[1] for item in token_and_distance)
            shortest_distance_tokens = [
                item[0] for item in token_and_distance if item[1] == min_distance
            ]
            return x, shortest_distance_tokens

def calculate_cooccurrence(chunk, tokenizer, window_size):
    # calculate the occurrence of words, (vocab_size, vocab_size)
    tknz_text = tokenizer(
        list(chunk),
        add_special_tokens=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tknz_text["input_ids"]
    vocab_size = len(tokenizer.get_vocab())
    co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for row in tqdm.tqdm(input_ids):
        for i, token_id in enumerate(row):
            for j in range(
                max(0, i - window_size), min(i + window_size, len(row))
            ):
                if i != j:
                    co_occurrence_matrix[token_id, row[j]] += 1
                    co_occurrence_matrix[row[j], token_id] += 1

    return co_occurrence_matrix

def tokenize_and_calculate_cooccurrence(
    text_corpus, tokenizer, window_size, n_processes
):
    # return (vocab_size, vocab_size) of tokenizer
    # ret
    with multiprocessing.Pool(n_processes) as pool:
        chunks = np.array_split(text_corpus, n_processes)
        co_occurrence_matrices = pool.starmap(
            calculate_cooccurrence,
            [(chunk, tokenizer, window_size) for chunk in chunks],
        )

    vocab_size = len(tokenizer.get_vocab())
    co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for matrix in co_occurrence_matrices:
        co_occurrence_matrix += matrix

    return co_occurrence_matrix

def align_vocab(
        base_model_name_or_path: str, # target
        blending_model_name_or_path: str, # source
        vocab_mapping_save_dir: str,
        cache_dir: str,
        model_max_length: int = 2048,
        vocab_mapping_type: str = 'default',
        # co-occurrence
        dataset_dir: str = '',
        window_size: int = 5,
        num_process: int = 1,
):
    logger.info(f"Map vocab from {os.path.basename(blending_model_name_or_path)} to {os.path.basename(base_model_name_or_path)}")

    if not os.path.exists(vocab_mapping_save_dir):
        os.makedirs(vocab_mapping_save_dir)

    logger.info(f"Save to: {vocab_mapping_save_dir}")

    base_tokenizer, _ = get_tokenizer(
        base_model_name_or_path, cache_dir, model_max_length
    )
    blending_tokenizer, _ = get_tokenizer(
        blending_model_name_or_path, cache_dir, model_max_length
    )

    base_tokens = list(base_tokenizer.get_vocab().keys())
    blending_tokens = list(blending_tokenizer.get_vocab().keys())
    # print(base_tokens)
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        base_tokenizer.__class__
    ]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_tokenizer.__class__
    ]

    if vocab_mapping_type == "default":

        logger.info("Mapping type: default")
        blending_to_base_mapping = dict()

        with multiprocessing.Pool(128) as pool:
            mapping_args = [
                (x, base_tokens, blending_model_special_token, base_model_special_token)
                for x in blending_tokens
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(find_best_mapping, mapping_args),
                    total=len(blending_tokens),
                )
            )

        for tmp_x, best_mapping in results:
            # key is the blending dict,
            blending_to_base_mapping[tmp_x] = best_mapping

    elif vocab_mapping_type == "co_occurrence":
        """
        1. Calculate basic one-to-many mapping based on editdistance
        2. Ranking based on co-occurrence to get one-to-one mapping
        This requires validation dataset...
        """
        # 1. Calculate basic one-to-many mapping based on editdistance
        blending_to_base_mapping = dict()

        with multiprocessing.Pool(64) as pool:
            mapping_args = [
                (x, base_tokens, blending_model_special_token, base_model_special_token)
                for x in blending_tokens
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(find_best_mapping, mapping_args),
                    total=len(blending_tokens),
                )
            )

        for tmp_x, best_mapping in results:
            blending_to_base_mapping[tmp_x] = best_mapping

        most_sim_blending_to_base_mapping = dict()
        with multiprocessing.Pool(64) as pool:
            mapping_args = [
                (
                    x,
                    base_tokens,
                    blending_model_special_token,
                    base_model_special_token,
                    False,
                )
                for x in blending_tokens
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(find_best_mapping, mapping_args),
                    total=len(blending_tokens),
                )
            )

        for tmp_x, basic_mapping in results:
            most_sim_blending_to_base_mapping[tmp_x] = basic_mapping

        dataset = load_from_disk(dataset_dir)[
            "validation"
        ]  # The training set is too large!

        # TODO: check this
        text_corpus = [dataset[i]["text"] for i in range(len(dataset))]  # texts

        base_model_co_occurrence_matrix = tokenize_and_calculate_cooccurrence(
            text_corpus, base_tokenizer, window_size, num_process
        )
        blending_model_co_occurrence_matrix = tokenize_and_calculate_cooccurrence(
            text_corpus, blending_tokenizer, window_size, num_process
        )
        base_vocab = base_tokenizer.get_vocab()  # vocab: id
        base_id_to_vocab = {v: k for k, v in base_vocab.items()}  # id: vocab
        blending_vocab = blending_tokenizer.get_vocab()  # vocab: id
        blending_id_to_vocab = {v: k for k, v in blending_vocab.items()}  # id: vocab
        # replace the special tokens
        tmp_blending_tokens = [
            x.replace(blending_model_special_token, base_model_special_token)
            for x in blending_tokens
        ]
        base_to_blending_mapping = {v: k for k, v in blending_to_base_mapping.items()}  # reverse
        blending_tokens_mapping_to_base_tokens = [
            blending_to_base_mapping[x] for x in tmp_blending_tokens
        ]  # tokens in both base & blending
        common_tokens = list(
            set(blending_tokens_mapping_to_base_tokens) & set(base_tokens)
        )  # pool
        blending_index = np.array(
            [
                blending_vocab[
                    base_to_blending_mapping[c].replace(
                        base_model_special_token, blending_model_special_token
                    )
                ]
                for c in common_tokens
            ]
        )
        base_index = np.array([base_vocab[c] for c in common_tokens])
        # filter occurrence matrix
        clip_blending_model_co_occurrence_matrix = blending_model_co_occurrence_matrix[
                                                   :, blending_index
                                                   ]
        clip_base_model_co_occurrence_matrix = base_model_co_occurrence_matrix[
                                               :, base_index
                                               ]

        def refined_mapping(key, value):
            if key in value:
                best_base_token = key  # best mapping
            elif len(value) == 1:
                best_base_token = value[0]  # only one mapping
            else:
                # if having more than one mapping
                blending_id = blending_vocab[
                    key.replace(base_model_special_token, blending_model_special_token)
                ]
                # !!! Use the co-occurrence to filter -> But why use different vector space to calculate the sim
                blending_vector = clip_blending_model_co_occurrence_matrix[blending_id]  # 1, n
                base_ids = np.array([base_vocab[base_token] for base_token in value])
                base_vectors = clip_base_model_co_occurrence_matrix[base_ids]  # k, n
                dot_product = np.dot(blending_vector, base_vectors.T)
                blending_norm = np.linalg.norm(blending_vector)
                base_norms = np.linalg.norm(base_vectors, axis=1)
                similarities = dot_product / (blending_norm * base_norms + 1e-6)
                assert len(similarities) == len(value)
                best_base_token = value[int(np.argmax(similarities))]
            return key, best_base_token

        updated_cnt = 0
        updated_dict = dict()
        for key, value in tqdm.tqdm(most_sim_blending_to_base_mapping.items()):
            if blending_to_base_mapping[key] == key:
                continue
            _, new_value = refined_mapping(key, value)
            if new_value != blending_to_base_mapping[key]:
                updated_cnt += 1
                updated_dict[key] = {
                    "default": blending_to_base_mapping[key],
                    "co-occurrence": new_value,
                }
                blending_to_base_mapping[key] = new_value

        logger.info(f"Co-occurrence updated {updated_cnt} tokens.")
        save_file = os.path.join(vocab_mapping_save_dir, 'map_vocab_updated_log.json')

        with open(save_file, 'w') as fout:
            json.dump(updated_dict, fout, indent=4)
    else:
        raise NotImplementedError

    # final target: get a map from blending to base, dict
    cnt = 0
    for k, v in blending_to_base_mapping.items():
        if k == v:
            cnt += 1
    logger.info(
        f"Total tokens in blending vocab: {len(blending_tokenizer.get_vocab())},"
        f"Total tokens in blending to base mapping: {len(blending_to_base_mapping)},"
        f"Total best matched tokens: {cnt}."
    )

    save_file = os.path.join(vocab_mapping_save_dir, 'map_vocab.json')
    with open(save_file, "w") as fout:
        json.dump(blending_to_base_mapping, fout)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mapping vocabs from different pretrain language models."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the base model.",
    )
    parser.add_argument(
        "--blending_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the blending model.",
    )
    parser.add_argument(
        "--vocab_mapping_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache dir."
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="The model max length."
    )
    parser.add_argument(
        "--vocab_mapping_type",
        type=str,
        default="default",
        help="The vocab mapping type.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="The local dir to load data."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="The window size to calculate co-occurrences.",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=1,
        help="The number of process."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")

    align_vocab(
        args.base_model_name_or_path,
        args.blending_model_name_or_path,
        args.vocab_mapping_save_dir,
        args.cache_dir,
        args.model_max_length,
        args.vocab_mapping_type,
        args.dataset_dir,
        args.window_size,
        args.num_process
    )