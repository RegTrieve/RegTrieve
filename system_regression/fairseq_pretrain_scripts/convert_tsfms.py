import argparse

import torch
from torch import nn

from transformers import Speech2TextConfig, Speech2TextForConditionalGeneration, Speech2TextConfig, Speech2TextTokenizer, Speech2TextProcessor,Speech2TextFeatureExtractor

# constant declaration
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"

def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        if "transformer_layers" in key:
            s_dict[key.replace("transformer_layers", "layers")] = s_dict.pop(key)
        elif "subsample" in key:
            s_dict[key.replace("subsample", "conv")] = s_dict.pop(key)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_fairseq_s2t_checkpoint_to_tfms(checkpoint_path, pytorch_dump_folder_path, model_size,vocab_file,sp_model):
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")
    args = m2m_100["args"]
    state_dict = m2m_100["model"]
    lm_head_weights = state_dict["decoder.output_projection.weight"]

    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    tie_embeds = True
    config_small = Speech2TextConfig(
        vocab_size=10000,
        max_source_positions=6000,
        max_target_positions=1024,
        encoder_layers=12,
        decoder_layers=6,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        d_model=256,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_function="relu",
        num_conv_layers=2,
        conv_channels=1024,
        conv_kernel_sizes=[5, 5],
        input_feat_per_channel=80,
        input_channels=1,
        tie_word_embeddings=True,  # tie_embeds
        num_beams=5,
        max_length=200,
        use_cache=True,
        decoder_start_token_id=2,
        early_stopping=True,
    )
    config_medium = Speech2TextConfig(
        vocab_size=10000,
        max_source_positions=6000,
        max_target_positions=1024,
        encoder_layers=12,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        d_model=512,
        dropout=0.15,
        attention_dropout=0.15,
        activation_dropout=0.15,
        activation_function="relu",
        num_conv_layers=2,
        conv_channels=1024,
        conv_kernel_sizes=[5, 5],
        input_feat_per_channel=80,
        input_channels=1,
        tie_word_embeddings=True,  # tie_embeds
        num_beams=5,
        max_length=200,
        use_cache=True,
        decoder_start_token_id=2,
        early_stopping=True,
    )
    config_large = Speech2TextConfig(
        vocab_size=10000,
        max_source_positions=6000,
        max_target_positions=1024,
        encoder_layers=12,
        decoder_layers=6,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        d_model=1024,
        dropout=0.2,
        attention_dropout=0.2,
        activation_dropout=0.2,
        activation_function="relu",
        num_conv_layers=2,
        conv_channels=1024,
        conv_kernel_sizes=[5, 5],
        input_feat_per_channel=80,
        input_channels=1,
        tie_word_embeddings=True,  # tie_embeds
        num_beams=5,
        max_length=200,
        use_cache=True,
        decoder_start_token_id=2,
        early_stopping=True,
    )
    if model_size=="small":
        config=config_small
    elif  model_size=="medium":
        config=config_medium
    elif model_size=="large":
        config=config_large
    else:
        raise ValueError("Unrecongnized model size! Which should be small,medium or large.")

    model = Speech2TextForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    if tie_embeds:
        model.lm_head = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.lm_head.weight.data = lm_head_weights

    model.save_pretrained(pytorch_dump_folder_path)

    # gen tokenizer & processor
    # Create and save tokenizer
    tokenizer = Speech2TextTokenizer(spm_file=sp_model,vocab_file=vocab_file,unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN)
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # Create and save processor
    feature_extractor = Speech2TextFeatureExtractor()
    processor = Speech2TextProcessor(tokenizer=tokenizer,feature_extractor=feature_extractor)
    processor.save_pretrained(pytorch_dump_folder_path)


    print("Successfully saved tfms model in",pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--fairseq_path", type=str, help="Path to the fairseq model (.pt) file.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--model_size", default="small", type=str,choices=["small","medium","large"], help="model size to be consistent with the arch arg in fairseq training")
    parser.add_argument("--vocab_file", type=str, help="the vocab file in fairseq")
    parser.add_argument("--sp_model", type=str, help="the sp model file in fairseq")
    args = parser.parse_args()
    print("fairseq_path:",args.fairseq_path,"\npytorch_dump_folder_path:",args.pytorch_dump_folder_path,"\nmodel_size:",args.model_size)
    convert_fairseq_s2t_checkpoint_to_tfms(args.fairseq_path,args.pytorch_dump_folder_path,args.model_size,args.vocab_file,args.sp_model)