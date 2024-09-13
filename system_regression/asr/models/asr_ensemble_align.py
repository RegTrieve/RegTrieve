# TODO: 1. implement average and maximum average (how to get the predict digit with asr model? should I modify generation method iteratively.)
# https://github.com/SebastianBodza/EnsembleForecasting/blob/main/EnsembleModel_torch.py
# 先尝试简单logit ensemble, 再尝试更加复杂的算法。We should get the predicted token distribution of LM and ensemble them. (参考Purifying Large Language Models by Ensembling a Small Language Model?, Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions In NLP Model Updates里的ensemble可能不具备参考价值，因为只是对最终的Logit进行ensemble)
# TODO:2. impplement dropout and perturb average

# 直接ensemble似乎是greedy decoding ensemble的概念，实际可能需要beam ensemble,然后算最后概率高的结果。直接ensemble的baseline是单模型greeedy decoding。
# 这里的logit ensemble只适合tokenizer一样的情况(small model-> large model, add model finetuning data, 不适合tokenizer变化的情况)

import torch
from torch import nn
import torch.nn.functional as F
from system_regression.asr.utils import wer_normalize_text, filter_eos_token, count_seq_end

from loguru import logger

from system_regression.asr.models.align_model import AlignModel

class ASRAlignEnsembleModel(nn.Module):
    """
    ASR ensemble with aligned model
    """
    def __init__(
            self,
            old_model_name,
            old_model,
            old_processor,
            new_model_name,
            new_model,
            new_processor,
            ensemble_method,
            weight_adj=False
    ):
        super(ASRAlignEnsembleModel, self).__init__()
        # old models
        self.old_model_name = old_model_name
        self.old_model = old_model
        self.old_processor = old_processor

        # new models
        self.new_model_name = new_model_name
        self.new_model = new_model
        self.new_processor = new_processor

        # others
        self.ensemble_method = ensemble_method
        self.weight_adj = weight_adj


        logger.debug(f'old model: {type(self.old_model)}')
        logger.debug(f'new model: {type(self.new_model)}')

    def forward_old_model(
            self,
            input_features,
            attention_mask,
            decoder_input_ids,
            decoder_mapped_input_ids
    ):
        batch_size = input_features.shape[0]

        with torch.no_grad():
            if isinstance(self.old_model, AlignModel):
                # original_decoder_input_ids
                # TODO: consider map terminal????
                decoder_input_ids, is_terminal, mapped_vocab_logits, decoder_mapped_input_ids, is_terminate_mapped = self.old_model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    mapped_input_ids=decoder_mapped_input_ids
                )
            else:
                output = self.old_model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask
                )
                mapped_vocab_logits = output.logits[:, -1, :]
                probs = F.softmax(mapped_vocab_logits, dim=-1)
                _, new_token_id = torch.max(probs, dim=-1)

                decoder_input_ids = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size, 1))], dim=-1)
                seq_end_count = count_seq_end(decoder_input_ids, self.old_processor.tokenizer.eos_token_id)

                is_terminal = seq_end_count == batch_size
                decoder_mapped_input_ids = None

        return decoder_input_ids, is_terminal, mapped_vocab_logits, decoder_mapped_input_ids

    def forward_new_model(
            self,
            input_features,
            attention_mask,
            decoder_input_ids,
            decoder_mapped_input_ids
    ):
        batch_size = input_features.shape[0]

        with torch.no_grad():
            if isinstance(self.new_model, AlignModel):
                # original_decoder_input_ids
                decoder_input_ids, is_terminal, mapped_vocab_logits, decoder_mapped_input_ids, is_terminate_mapped = self.new_model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    mapped_input_ids=decoder_mapped_input_ids
                )
            else:
                output = self.new_model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask
                )

                mapped_vocab_logits = output.logits[:, -1, :]
                probs = F.softmax(mapped_vocab_logits, dim=-1)
                _, new_token_id = torch.max(probs, dim=-1)

                decoder_input_ids = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size, 1))], dim=-1)
                seq_end_count = count_seq_end(decoder_input_ids, self.new_processor.tokenizer.eos_token_id)

                is_terminal = seq_end_count == batch_size

                decoder_mapped_input_ids = None

        return decoder_input_ids, is_terminal, mapped_vocab_logits, decoder_mapped_input_ids

    def forward(
            self,
            old_input_features,
            old_attention_mask,
            old_decoder_mapped_input_ids,
            old_ensemble_decoder_input_ids,
            new_input_features,
            new_attention_mask,
            new_decoder_mapped_input_ids,
            new_ensemble_decoder_input_ids,
    ):
        # TODO: now support old to new

        # keep input arguments
        i_old_input_features = old_input_features
        i_old_attention_mask = old_attention_mask
        i_old_decoder_mapped_input_ids = old_decoder_mapped_input_ids
        i_old_ensemble_decoder_input_ids = old_ensemble_decoder_input_ids
        i_new_input_features = new_input_features
        i_new_attention_mask = new_attention_mask
        i_new_decoder_mapped_input_ids = new_decoder_mapped_input_ids
        i_new_ensemble_decoder_input_ids = new_ensemble_decoder_input_ids

        # update with original ids
        # old self,
        (old_decoder_input_ids,
         old_model_is_terminal,
         old_model_mapped_vocab_logits,
         old_decoder_mapped_input_ids) = self.forward_old_model(
            input_features=old_input_features,
            attention_mask=old_attention_mask,
            decoder_input_ids=old_ensemble_decoder_input_ids, # old_decoder_input_ids,  # TODO: change to the ensemble
            decoder_mapped_input_ids=old_decoder_mapped_input_ids
        )

        # update with ensemble ids
        (new_decoder_input_ids,
         new_model_is_terminal,
         new_model_mapped_vocab_logits,
         new_decoder_mapped_input_ids) = self.forward_new_model(
            input_features=new_input_features,
            attention_mask=new_attention_mask,
            decoder_input_ids=new_ensemble_decoder_input_ids, # TODO: change to the ensemble
            decoder_mapped_input_ids=new_decoder_mapped_input_ids
        )

        # 最后一个logit才是推理输出的logit，前面都是context logit
        logits1 = old_model_mapped_vocab_logits # F.softmax(old_model_mapped_vocab_logits, dim=-1)  # outputs1.logits[:, -1, :]
        logits2 = new_model_mapped_vocab_logits # F.softmax(new_model_mapped_vocab_logits, dim=-1)  # outputs2.logits[:, -1, :]

        # logger.debug('logits1 shape: {}, logits2 shape: {}'.format(logits1.shape, logits2.shape))

        batch_size = old_input_features.shape[0]

        # exit()
        new_token_id = None
        if self.ensemble_method == 'max':
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            merged_prob = torch.max(probs1, probs2)
            # TODO: 添加sampling generation的方式，先取max，再sample
            max_prob1, max_indices1 = torch.max(merged_prob, dim=-1)
            # max_prob2, max_indices2 = torch.max(probs2, dim=-1)
            new_token_id = max_indices1
        elif self.ensemble_method == 'avg':
            merge_logits = (logits1 + logits2) / 2
            probs = F.softmax(merge_logits, dim=-1)
            _, new_token_id = torch.max(probs, dim=-1)
        elif self.ensemble_method == 'dropout':
            # TODO: not fix
            loops = 10
            # old_confs = F.softmax(output1, dim=-1)
            # new_confs = F.softmax(output2, dim=-1)
            pred1 = []
            pred2 = []
            # 这里设置为train就将dropout打开了
            self.old_model.train()
            self.new_model.train()
            for _ in range(loops):
                (_, _, e_old_model_mapped_vocab_logits, _) = self.forward_old_model(
                    input_features=i_old_input_features,
                    attention_mask=i_old_attention_mask,
                    decoder_input_ids=i_old_ensemble_decoder_input_ids,
                    # old_decoder_input_ids,  # TODO: change to the ensemble
                    decoder_mapped_input_ids=i_old_decoder_mapped_input_ids
                )
                pred1.append(F.softmax(e_old_model_mapped_vocab_logits, dim=-1))

                (_, _, e_new_model_mapped_vocab_logits, _) = self.forward_new_model(
                    input_features=i_new_input_features,
                    attention_mask=i_new_attention_mask,
                    decoder_input_ids=i_new_ensemble_decoder_input_ids,
                    # new_decoder_input_ids,  # TODO: change to the ensemble
                    decoder_mapped_input_ids=i_new_decoder_mapped_input_ids
                )
                pred2.append(F.softmax(e_new_model_mapped_vocab_logits, dim=-1))

                # pred1.append(F.softmax(self.model1(input_features, decoder_input_ids=decoder_input_ids,
                #                                    attention_mask=attention_mask).logits[:, -1, :], dim=-1))
                # pred2.append(F.softmax(self.model2(input_features, decoder_input_ids=decoder_input_ids,
                #                                    attention_mask=attention_mask).logits[:, -1, :], dim=-1))
            var_pred1 = torch.zeros(logits1.shape).cuda()
            var_pred2 = torch.zeros(logits2.shape).cuda()
            mean_pred1 = torch.zeros(logits1.shape).cuda()
            mean_pred2 = torch.zeros(logits2.shape).cuda()
            for i in range(loops):
                mean_pred1 += pred1[i]
                mean_pred2 += pred2[i]
                var_pred1 += pred1[i].square()
                var_pred2 += pred2[i].square()
            var_pred1 = (var_pred1 / loops) - (mean_pred1 / loops).square()
            var_pred2 = (var_pred2 / loops) - (mean_pred2 / loops).square()
            # clip for numerical stability
            weight1 = torch.clamp(var_pred1.sum(dim=-1), min=1e-3, max=1 - 1e-3)
            weight2 = torch.clamp(var_pred2.sum(dim=-1), min=1e-3, max=1 - 1e-3)
            # add a priori with weight1=0.0, weight2=1.0.
            if self.weight_adj:
                w1 = torch.unsqueeze((weight2 / (2 * (weight1 + weight2)) + 0.0), -1)
                w2 = torch.unsqueeze((weight1 / (2 * (weight1 + weight2)) + 0.5), -1)
            else:
                w1 = torch.unsqueeze((weight2 / (weight1 + weight2)), -1)
                w2 = torch.unsqueeze((weight1 / (weight1 + weight2)), -1)
            probs = torch.mul(mean_pred1 / loops, w1) + torch.mul(mean_pred2 / loops, w2)
            _, new_token_id = torch.max(probs, dim=-1)
            self.old_model.eval()
            self.new_model.eval()

        elif self.ensemble_method == 'pertub':
            loops = 10
            # old_confs = F.softmax(output1, dim=-1)
            # new_confs = F.softmax(output2, dim=-1)
            pred1 = []
            pred2 = []
            for _ in range(loops):
                pert = torch.randn(i_old_input_features.shape).cuda()
                (_, _, e_old_model_mapped_vocab_logits, _) = self.forward_old_model(
                    input_features=i_old_input_features + 0.05 * pert,
                    attention_mask=i_old_attention_mask,
                    decoder_input_ids=i_old_ensemble_decoder_input_ids,
                    # old_decoder_input_ids,  # TODO: change to the ensemble
                    decoder_mapped_input_ids=i_old_decoder_mapped_input_ids
                )
                pred1.append(F.softmax(e_old_model_mapped_vocab_logits, dim=-1))

                pert = torch.randn(i_new_input_features.shape).cuda()
                (_, _, e_new_model_mapped_vocab_logits, _) = self.forward_new_model(
                    input_features=i_new_input_features + 0.05 * pert,
                    attention_mask=i_new_attention_mask,
                    decoder_input_ids=i_new_ensemble_decoder_input_ids,
                    # new_decoder_input_ids,  # TODO: change to the ensemble
                    decoder_mapped_input_ids=i_new_decoder_mapped_input_ids
                )
                pred2.append(F.softmax(e_new_model_mapped_vocab_logits, dim=-1))

                #
                # pred1.append(F.softmax(self.model1(input_features + 0.05 * pert, decoder_input_ids=decoder_input_ids,
                #                                    attention_mask=attention_mask).logits[:, -1, :], dim=-1))
                # # pert = torch.randn(x.shape).cuda()
                # pred2.append(F.softmax(self.model2(input_features + 0.05 * pert, decoder_input_ids=decoder_input_ids,
                #                                    attention_mask=attention_mask).logits[:, -1, :], dim=-1))
            var_pred1 = torch.zeros(logits1.shape).cuda()
            var_pred2 = torch.zeros(logits2.shape).cuda()
            mean_pred1 = torch.zeros(logits1.shape).cuda()
            mean_pred2 = torch.zeros(logits2.shape).cuda()
            for _ in range(loops):
                mean_pred1 += pred1[_]
                mean_pred2 += pred2[_]
                var_pred1 += pred1[_].square()
                var_pred2 += pred2[_].square()
            var_pred1 = (var_pred1 / loops) - (mean_pred1 / loops).square()
            var_pred2 = (var_pred2 / loops) - (mean_pred2 / loops).square()
            # clip for numerical stability
            weight1 = torch.clamp(var_pred1.sum(dim=-1), min=1e-5, max=1 - 1e-5)
            weight2 = torch.clamp(var_pred2.sum(dim=-1), min=1e-5, max=1 - 1e-5)
            # set priori weight as w1=0.0, w2=1.0
            if self.weight_adj:
                w1 = torch.unsqueeze((weight2 / (2 * (weight1 + weight2)) + 0.0), -1)
                w2 = torch.unsqueeze((weight1 / (2 * (weight1 + weight2)) + 0.5), -1)
            else:
                w1 = torch.unsqueeze((weight2 / (weight1 + weight2)), -1)
                w2 = torch.unsqueeze((weight1 / (weight1 + weight2)), -1)
            probs = torch.mul(mean_pred1 / loops, w1) + torch.mul(mean_pred2 / loops, w2)
            _, new_token_id = torch.max(probs, dim=-1)


        # obtain ensemble ids for new model
        if isinstance(self.new_model, AlignModel):
            # map new to old
            new_ensemble_token_id = self.new_model.inverse_map_token(new_token_id)
        else:
            new_ensemble_token_id = new_token_id
        new_ensemble_decoder_input_ids = torch.cat([
            new_ensemble_decoder_input_ids,
            torch.reshape(new_ensemble_token_id, (batch_size, 1))
        ], dim=-1)
        new_seq_end_count = count_seq_end(new_ensemble_decoder_input_ids, self.new_processor.tokenizer.eos_token_id)
        new_is_terminal = new_seq_end_count == batch_size

        # obtain ensemble ids for old model
        if isinstance(self.old_model, AlignModel):
            # map old to new
            old_ensemble_token_id = self.old_model.reverse_map_token(new_token_id)
        else:
            old_ensemble_token_id = new_token_id
        old_ensemble_decoder_input_ids = torch.cat([
            old_ensemble_decoder_input_ids,
            torch.reshape(old_ensemble_token_id, (batch_size, 1))
        ], dim=-1)
        old_seq_end_count = count_seq_end(old_ensemble_decoder_input_ids, self.old_processor.tokenizer.eos_token_id)
        old_is_terminal = old_seq_end_count == batch_size

        return (
            new_ensemble_decoder_input_ids, # new space -> convert to old space (TODO)
            new_decoder_mapped_input_ids,
            new_is_terminal,
            old_ensemble_decoder_input_ids, # old space -> convert to new space (TODO)
            old_decoder_mapped_input_ids,
            old_is_terminal
        )

    def generate_sample_batch(
            self,
            old_input_features,
            new_input_features,
            old_attention_mask=None,
            new_attention_mask=None,
            max_length=200
    ):
        # greedy generation now, may use other generation methods like beam search
        assert new_input_features.shape[0] == old_input_features.shape[0]

        batch_size = old_input_features.shape[0]
        ###### initialize ######
        # obtain old
        eos_token_id_old = self.old_processor.tokenizer.eos_token_id
        if self.old_model_name == 's2t':
            # old model is s2t
            old_ensemble_decoder_input_ids = torch.tensor(
                [batch_size * [eos_token_id_old]]
            ).reshape(batch_size, -1).to(self.old_model.device)
        else:
            # old model is whisper
            old_ensemble_decoder_input_ids = torch.tensor(
                [batch_size * [self.old_model.generation_config.decoder_start_token_id]]
            ).reshape(batch_size, -1).to(self.old_model.device)

        # obtain new
        eos_token_id_new = self.new_processor.tokenizer.eos_token_id
        if self.new_model_name == 's2t':
            # new model is s2t
            new_ensemble_decoder_input_ids = torch.tensor(
                [batch_size * [eos_token_id_new]]
            ).reshape(batch_size, -1).to(self.new_model.device)
        else:
            # new model is whisper
            new_ensemble_decoder_input_ids = torch.tensor(
                [batch_size * [self.new_model.generation_config.decoder_start_token_id]]
            ).reshape(batch_size, -1).to(self.new_model.device)


        # get mapped ids
        if isinstance(self.old_model, AlignModel):
            old_decoder_mapped_input_ids = new_ensemble_decoder_input_ids
        else:
            old_decoder_mapped_input_ids = None

        if isinstance(self.new_model, AlignModel):
            new_decoder_mapped_input_ids = old_ensemble_decoder_input_ids
        else:
            new_decoder_mapped_input_ids = None

        for _ in range(max_length):
            new_ensemble_decoder_input_ids, new_decoder_mapped_input_ids, new_is_terminal, old_ensemble_decoder_input_ids, old_decoder_mapped_input_ids, old_is_terminal = self.forward(
                old_input_features=old_input_features,
                old_attention_mask=old_attention_mask,
                old_decoder_mapped_input_ids=old_decoder_mapped_input_ids,
                old_ensemble_decoder_input_ids=old_ensemble_decoder_input_ids,
                new_input_features=new_input_features,
                new_attention_mask=new_attention_mask,
                new_decoder_mapped_input_ids=new_decoder_mapped_input_ids,
                new_ensemble_decoder_input_ids=new_ensemble_decoder_input_ids
            )

            if new_is_terminal or old_is_terminal:
                # TODO: check the terminal conditions
                break

        # TODO: only support old to new
        if self.new_model_name == 'whisper':
            new_ensemble_decoder_input_ids[new_ensemble_decoder_input_ids == -100] = self.new_processor.tokenizer.pad_token_id
        new_ensemble_decoder_input_ids = filter_eos_token(new_ensemble_decoder_input_ids, self.new_processor.tokenizer.eos_token_id)
        return self.new_processor.batch_decode(new_ensemble_decoder_input_ids, skip_special_tokens=True)