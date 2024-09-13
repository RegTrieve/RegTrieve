import os
import json
import torch

from loguru import logger
from tqdm import tqdm
from typing import Optional
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from system_regression.asr.align.align_utils import get_tokenizer
from system_regression.common.const import Constants
from system_regression.asr.models.align_model import AlignModel
from system_regression.asr.models.asr_ensemble_align import ASRAlignEnsembleModel
from system_regression.asr.utils import load_model, wer_normalize_text
from system_regression.data_prep.heysquad import get_heysquad_datasets

class MergedIterableDataset(IterableDataset):
    def __init__(self, dataset_old, dataset_new):
        super(MergedIterableDataset, self).__init__()
        self.dataset_old = dataset_old
        self.dataset_new = dataset_new

    def __iter__(self):
        iterator_old = iter(self.dataset_old)
        iterator_new = iter(self.dataset_new)
        for batch_old, batch_new in zip(iterator_old, iterator_new):
            merged_batch = {
                'old': batch_old, # model
                'new': batch_new # base
            }
            yield merged_batch

class MergedCollector:

    def __init__(
            self,
            collector_old,
            collector_new
    ):
        self.collector_old = collector_old
        self.collector_new = collector_new

    def __call__(self, batch):

        batch_old = []
        batch_new = []
        for item in batch:
            batch_old.append(item['old'])
            batch_new.append(item['new'])
        batch_old = self.collector_old(batch_old)
        batch_new = self.collector_new(batch_new)
        return batch_old, batch_new

class AlignWrapper:

    def __init__(
            self,
            use_align: bool,
            model_name: str,
            model_path: Optional[str],
            base_model_name: str,
            base_model_path: Optional[str],
            vocab_mapper_path: Optional[str],
            reverse_vocab_mapper_path: Optional[str]
    ):

        self.use_align = use_align
        self.model_name = model_name
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.base_model_path = base_model_path
        self.vocab_mapper_path = vocab_mapper_path
        self.reverse_vocab_mapper_path = reverse_vocab_mapper_path

        self.tokenizer = None
        self.base_tokenizer = None
        self.vocab_mapper = None
        self.reverse_vocab_mapper = None

        # inner parameters
        self.model = None
        self.processor = None
        self.data_collator = None

        self.base_model = None
        self.base_processor = None
        self.base_data_collator = None

        self.align_model = None

        if self.use_align:
            with open(self.vocab_mapper_path, 'r') as f:
                self.vocab_mapper = json.load(f)

            with open(self.reverse_vocab_mapper_path, 'r') as f:
                self.reverse_vocab_mapper = json.load(f)

            # source
            self.tokenizer, _ = get_tokenizer(self.model_path, "", 2048)
            self.base_tokenizer, _ = get_tokenizer(self.base_model_path, "", 2048)

    def create_align_model(self, debug=True) -> AlignModel:
        # create model
        self.model, self.processor, self.data_collator = load_model(
            self.model_name,
            self.model_path,
            back_model_path=self.base_model_path
        )
        self.model = self.model.to(Constants.device)
        self.model.eval()

        # create base model
        self.base_model, self.base_processor, self.base_data_collator = load_model(
            self.base_model_name,
            self.base_model_path,
            back_model_path=self.model_path # TODO: check this
        )
        self.base_model = self.base_model.to(Constants.device)
        self.base_model.eval()

        self.align_model = AlignModel(
            self.model,
            self.tokenizer,
            self.base_model,
            self.base_tokenizer,
            self.vocab_mapper,
            self.reverse_vocab_mapper,
            10,
            debug=debug
        )

        self.align_model = self.align_model.to(Constants.device)
        self.align_model.eval()

    @classmethod
    def load_from_config(cls, opt):
        old_path = os.path.join(Constants.pretrained_model_dir, opt.oldmodel_path)
        new_path = os.path.join(Constants.pretrained_model_dir, opt.newmodel_path)

        vocab_folder = opt.vocab_path
        if opt.align == 'old2new':
            use_align = True
            vocab_path = os.path.join(vocab_folder, f"{opt.oldmodel_path}_to_{opt.newmodel_path}/map_vocab.json")
            reverse_vocab_path = os.path.join(vocab_folder,
                                              f"{opt.newmodel_path}_to_{opt.oldmodel_path}/map_vocab.json")
            model_name = opt.old_model
            model_path = old_path
            base_model_name = opt.new_model
            base_model_path = new_path
        elif opt.align == 'new2old':
            use_align = True
            vocab_path = os.path.join(vocab_folder, f"{opt.newmodel_path}_to_{opt.oldmodel_path}/map_vocab.json")
            reverse_vocab_path = os.path.join(vocab_folder,
                                              f"{opt.oldmodel_path}_to_{opt.newmodel_path}/map_vocab.json")
            model_name = opt.new_model
            model_path = new_path
            base_model_name = opt.old_model
            base_model_path = old_path
        else:
            use_align = False
            vocab_path = None
            reverse_vocab_path = None
            model_name = None
            model_path = None
            base_model_name = None
            base_model_path = None

        return cls(
            use_align=use_align,
            model_name=model_name,
            model_path=model_path,
            base_model_name=base_model_name,
            base_model_path=base_model_path,
            vocab_mapper_path=vocab_path,
            reverse_vocab_mapper_path=reverse_vocab_path
        )

    def test_single(self, dev_ref_ids, human_machine, dev_ref_hashmap, opt, debug = True):
        self.create_align_model(debug)

        self.align_model.eval()

        # get input dataloader
        # original
        _, _, test_set, test_val_set = get_heysquad_datasets(
            self.processor,
            model=self.model_name,
            ref_dev_ids=dev_ref_ids,
            ref_train_ids=None,
            debugging=False,
            test_only=True,
            human_machine=human_machine,
            ref_dev_hashmap=dev_ref_hashmap
        )
        # mapped
        _, _, base_test_set, base_test_val_set = get_heysquad_datasets(
            self.base_processor,
            model=self.base_model_name,
            ref_dev_ids=dev_ref_ids,
            ref_train_ids=None,
            debugging=False,
            test_only=True,
            human_machine=human_machine,
            ref_dev_hashmap=dev_ref_hashmap
        )

        # merge this two dataset
        test_set = MergedIterableDataset(
            dataset_old=test_set,
            dataset_new=base_test_set
        )
        data_collator = MergedCollector(
            collector_old=self.data_collator,
            collector_new=self.base_data_collator
        )

        test_loader = DataLoader(
            test_set,
            batch_size=opt.batch_size,
            shuffle=False,
            collate_fn=data_collator
        )

        wers = {}
        predictions = []
        refs = []
        ids = []
        res = {}
        max_len = Constants.max_question_len

        all_logging_data = {}  # q_id: logging_data
        with torch.no_grad():
            for data, base_data in tqdm(test_loader):
                input_features = data["input_features"].to(Constants.device)
                attention_mask = data['attention_mask'].to(Constants.device) if 'attention_mask' in data else None

                mapped_transcription_asr, original_transcription_asr = self.align_model.generate_sample_batch(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    max_length=max_len
                )

                logger.debug(f"Original: {original_transcription_asr}")
                logger.debug(f"Mapped: {mapped_transcription_asr}")

                if self.base_model_name == 'whisper':
                    questions = self.base_processor.batch_decode(base_data['labels'], skip_special_tokens=True)
                elif self.base_model_name == 's2t':
                    questions = base_data['questions']

                transcription_asr = mapped_transcription_asr

                # add eval logics
                transcription_asr = wer_normalize_text(transcription_asr)
                predictions.extend(transcription_asr)
                questions = wer_normalize_text(questions)
                for idx, _ in enumerate(data["ids"]):
                    res[_] = {
                        'question': questions[idx],
                        # 'context':data["context"][0],
                        'transcribe': transcription_asr[idx],
                    }

                refs.extend(questions)
                ids.extend(data["ids"])

            # if len(all_logging_data) > 0:
            #     decode_logging_data(all_logging_data, processor.tokenizer)

            wer_thresh = 0.2
            err_count = 0
            empty_ref_ids = []
            for i in range(len(ids)):
                if len(refs[i]) == 0:
                    empty_ref_ids.append(ids[i])
                    wers[ids[i]] = 1
                else:
                    wers[ids[i]] = Constants.wer.compute(predictions=[predictions[i]], references=[refs[i]])
                res[ids[i]]['wer'] = wers[ids[i]]
                if wers[ids[i]] > wer_thresh:
                    err_count += 1
            # avg_wer = Constants.wer.compute(predictions=predictions, references=refs)
            avg_wer = sum(list(wers.values())) / len(wers)
            # TODO: add variance of wer and histogram
            logger.info('Avg wer %f, Err rate %f' % (avg_wer, err_count / len(refs)))
            logger.info('Empty Ref id num: %d' % len(empty_ref_ids))
        # if return_logging:
        #     return wers, res, all_logging_data
        # else:

        return wers, res

    def test_ensemble(self, dev_ref_ids, human_machine, dev_ref_hashmap, opt, debug = True):
        # create ensemble model
        self.create_align_model(debug)
        ensemble_model = ASRAlignEnsembleModel(
            old_model_name=self.model_name,
            old_model=self.align_model, # NOTE: this is align model
            old_processor=self.processor,
            new_model_name=self.base_model_name,
            new_model=self.base_model,
            new_processor=self.base_processor,
            ensemble_method=opt.ensemble,
            weight_adj=opt.weight_adj
        )
        ensemble_model.eval()

        # create dataloader
        _, _, test_set, test_val_set = get_heysquad_datasets(
            self.processor,
            model=self.model_name,
            ref_dev_ids=dev_ref_ids,
            ref_train_ids=None,
            debugging=False,
            test_only=True,
            human_machine=human_machine,
            ref_dev_hashmap=dev_ref_hashmap
        )
        # mapped
        _, _, base_test_set, base_test_val_set = get_heysquad_datasets(
            self.base_processor,
            model=self.base_model_name,
            ref_dev_ids=dev_ref_ids,
            ref_train_ids=None,
            debugging=False,
            test_only=True,
            human_machine=human_machine,
            ref_dev_hashmap=dev_ref_hashmap
        )

        # merge this two dataset
        test_set = MergedIterableDataset(
            dataset_old=test_set,
            dataset_new=base_test_set
        )
        data_collator = MergedCollector(
            collector_old=self.data_collator,
            collector_new=self.base_data_collator
        )

        test_loader = DataLoader(
            test_set,
            batch_size=opt.batch_size,
            shuffle=False,
            collate_fn=data_collator
        )

        wers = {}
        predictions = []
        refs = []
        ids = []
        res = {}
        max_len = Constants.max_question_len

        all_logging_data = {}  # q_id: logging_data
        with torch.no_grad():
            for data, base_data in tqdm(test_loader):
                old_input_features = data["input_features"].to(Constants.device)
                old_attention_mask = data['attention_mask'].to(Constants.device) if 'attention_mask' in data else None

                new_input_features = base_data["input_features"].to(Constants.device)
                new_attention_mask = base_data['attention_mask'].to(Constants.device) if 'attention_mask' in base_data else None

                transcription_asr = ensemble_model.generate_sample_batch(
                    old_input_features=old_input_features,
                    new_input_features=new_input_features,
                    old_attention_mask=old_attention_mask,
                    new_attention_mask=new_attention_mask,
                    max_length=max_len
                )

                logger.debug(f"transcription_asr: {transcription_asr}")

                if self.base_model_name == 'whisper':
                    questions = self.base_processor.batch_decode(base_data['labels'], skip_special_tokens=True)
                elif self.base_model_name == 's2t':
                    questions = base_data['questions']

                # add eval logics
                transcription_asr = wer_normalize_text(transcription_asr)
                predictions.extend(transcription_asr)
                questions = wer_normalize_text(questions)
                for idx, _ in enumerate(data["ids"]):
                    res[_] = {
                        'question': questions[idx],
                        # 'context':data["context"][0],
                        'transcribe': transcription_asr[idx],
                    }

                refs.extend(questions)
                ids.extend(data["ids"])

            # if len(all_logging_data) > 0:
            #     decode_logging_data(all_logging_data, processor.tokenizer)

            wer_thresh = 0.2
            err_count = 0
            empty_ref_ids = []
            for i in range(len(ids)):
                if len(refs[i]) == 0:
                    empty_ref_ids.append(ids[i])
                    wers[ids[i]] = 1
                else:
                    wers[ids[i]] = Constants.wer.compute(predictions=[predictions[i]], references=[refs[i]])
                res[ids[i]]['wer'] = wers[ids[i]]
                if wers[ids[i]] > wer_thresh:
                    err_count += 1
            # avg_wer = Constants.wer.compute(predictions=predictions, references=refs)
            avg_wer = sum(list(wers.values())) / len(wers)
            # TODO: add variance of wer and histogram
            logger.info('Avg wer %f, Err rate %f' % (avg_wer, err_count / len(refs)))
            logger.info('Empty Ref id num: %d' % len(empty_ref_ids))
        # if return_logging:
        #     return wers, res, all_logging_data
        # else:

        return wers, res