import json
import logging
import os
from dataclasses import dataclass, field
from itertools import accumulate
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed

from dataset import DataArguments, GodasDataset
from model import ModelArguments, ORCADLConfig, ORCADLModel
from trainer import TrainingArguments, collate_fn, setup_logger


logger = logging.getLogger(__name__)


def add_months(time_str: str, months: int) -> str:
    year, month = map(int, time_str.split("_"))
    month_index = (year * 12 + (month - 1)) + months
    new_year, new_month_idx = divmod(month_index, 12)
    return f"{new_year}_{new_month_idx + 1}"


def resolve_subset_indices(length: int, num_samples: Optional[int], indices: List[int]) -> Optional[List[int]]:
    if num_samples is None and len(indices) == 0:
        return None

    if len(indices) > 0:
        if len(indices) == 1:
            end = indices[0]
            if end < 0:
                end = length + end + 1
            end = min(end, length)
            return list(range(end))

        start, end = indices
        if end < 0:
            end = length + end + 1
        start = max(start, 0)
        end = min(end, length)
        return list(range(start, end))

    return list(range(min(num_samples, length)))


class GodasAllInitDataset(GodasDataset):
    def __init__(self, args, init_start_year: Optional[int] = 2011, init_end_year: Optional[int] = 2023) -> None:
        super().__init__(args)
        self.init_indices = self._build_init_indices(init_start_year, init_end_year)
        self.init_times = [self.times[i + self.input_steps - 1] for i in self.init_indices]
        self.num_samples = len(self.init_indices)

    def _build_init_indices(self, init_start_year: Optional[int], init_end_year: Optional[int]) -> List[int]:
        last_valid_index = self.num_times - self.input_steps
        if last_valid_index < 0:
            raise ValueError("Not enough time steps for the configured input_steps.")

        init_indices = []
        for idx in range(last_valid_index + 1):
            init_time = self.times[idx + self.input_steps - 1]
            init_year = int(init_time.split("_")[0])
            if init_start_year is not None and init_year < init_start_year:
                continue
            if init_end_year is not None and init_year > init_end_year:
                continue
            init_indices.append(idx)

        if len(init_indices) == 0:
            raise ValueError("No valid initialization samples found for the requested year range.")

        return init_indices

    def __len__(self):
        return self.num_samples

    def __getitem__(self, subset_index):
        index = self.init_indices[subset_index]
        start_month = int(self.times[index].split("_")[-1])

        inputs_range = range(index, index + self.input_steps)
        ocean_vars = self.get_ocean_vars(self.root, inputs_range)
        atmo_vars = self.get_atmo_vars(self.root, inputs_range)

        return {
            "ocean_vars": ocean_vars,
            "atmo_vars": atmo_vars,
            "month": torch.tensor(start_month),
        }


@dataclass
class PredictAllArguments(TrainingArguments):
    num_test_samples: Optional[int] = field(
        default=None, metadata={"help": "How many samples are used for prediction after year filtering."}
    )
    test_data_indices: List[int] = field(
        default_factory=list,
        metadata={"help": "Subset indices in filtered prediction samples. Supports [end] or [start, end)."},
    )
    save_preds: bool = field(
        default=False, metadata={"help": "If True, save model predictions."}
    )
    ckpt_list: List[str] = field(
        default_factory=list
    )
    config_path_list: List[str] = field(
        default_factory=list
    )
    save_vars: List[str] = field(
        default_factory=list
    )
    init_start_year: Optional[int] = field(
        default=2011, metadata={"help": "Inclusive initialization start year (default GODAS test period: 2011-2023)."}
    )
    init_end_year: Optional[int] = field(
        default=2023, metadata={"help": "Inclusive initialization end year (default GODAS test period: 2011-2023)."}
    )

    def __post_init__(self):
        super().__post_init__()
        if len(self.test_data_indices) > 2:
            raise ValueError("Indices should contain at most two values: [end] or [start, end).")


def build_models(testing_args: PredictAllArguments, data_args: DataArguments, model_args: ModelArguments, var_list, var_index):
    model_list = []

    if len(testing_args.ckpt_list) == 0:
        raise ValueError("ckpt_list cannot be empty.")

    if len(testing_args.ckpt_list) != len(testing_args.config_path_list):
        if len(testing_args.config_path_list) == 1:
            testing_args.config_path_list = [testing_args.config_path_list[0]] * len(testing_args.ckpt_list)
            logger.warning("Only one config path is provided, using it for all checkpoints.")
        else:
            raise ValueError("config_path_list should have the same length as ckpt_list, or length 1.")

    for ckpt_path, cfg_path in zip(testing_args.ckpt_list, testing_args.config_path_list):
        config = ORCADLConfig.from_json_file(cfg_path)
        config.update_from_args(model_args)

        model = ORCADLModel(config)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

        if model.config.var_list != var_list or model.config.var_index != var_index:
            raise ValueError("var_list/var_index in args is not the same as in pretrained model config")

        model.config.update({
            "predict_time_steps": data_args.predict_steps,
        })
        model_list.append(model)

    return model_list


def save_prediction_metadata(outdir: str, init_times: List[str], data_times: List[str], input_steps: int, predict_steps: int):
    metadata = {
        "mode": "all_available_initializations",
        "input_steps": input_steps,
        "predict_steps": predict_steps,
        "num_data_months": len(data_times),
        "num_init_samples": len(init_times),
        "data_start_time": data_times[0] if data_times else None,
        "data_end_time": data_times[-1] if data_times else None,
        "first_init_time": init_times[0] if init_times else None,
        "last_init_time": init_times[-1] if init_times else None,
        "step_time_ranges": {},
    }

    if init_times:
        for step in range(1, predict_steps + 1):
            metadata["step_time_ranges"][f"step{step}"] = {
                "first_target_time": add_months(init_times[0], step),
                "last_target_time": add_months(init_times[-1], step),
            }

    with open(os.path.join(outdir, "predict_all_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = HfArgumentParser((PredictAllArguments, DataArguments, ModelArguments))
    testing_args, data_args, model_args = parser.parse_args_into_dataclasses()

    setup_logger(testing_args, logger)
    logger.info(f"Prediction parameters {testing_args}")

    set_seed(testing_args.seed)

    predict_dataset = GodasAllInitDataset(
        data_args,
        init_start_year=testing_args.init_start_year,
        init_end_year=testing_args.init_end_year,
    )

    subset_indices = resolve_subset_indices(
        len(predict_dataset),
        testing_args.num_test_samples,
        list(testing_args.test_data_indices),
    )
    if subset_indices is not None:
        predict_dataset.init_indices = [predict_dataset.init_indices[i] for i in subset_indices]
        predict_dataset.init_times = [predict_dataset.init_times[i] for i in subset_indices]
        predict_dataset.num_samples = len(predict_dataset.init_indices)

    var_list = predict_dataset.get_input_var_list_cmip6()
    var_index = [predict_dataset.get_var_index(v) for v in var_list]
    model_list = build_models(testing_args, data_args, model_args, var_list, var_index)

    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=testing_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=testing_args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in model_list:
        model.to(device)
        model.eval()

    preds = []
    model_config = model_list[0].config

    outdir = testing_args.output_dir
    preds_save_dir = os.path.join(outdir, "preds")
    os.makedirs(outdir, exist_ok=True)
    if testing_args.save_preds:
        os.makedirs(preds_save_dir, exist_ok=True)

    num_model = len(model_list)

    for batch in tqdm(dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.inference_mode():
            logits_sum = 0.0
            for model in model_list:
                logits_sum += model(**batch).preds
            logits = logits_sum / num_model

        preds.append(logits.detach().cpu())

    split_indices = list(accumulate(model_config.out_chans))[:-1]
    split_axis = 1 if data_args.predict_steps == 1 else 2
    preds = [torch.tensor_split(p, split_indices, dim=split_axis) for p in preds]

    all_preds = []
    for i in range(len(preds[0])):
        all_preds.append(torch.cat([p[i] for p in preds], dim=0))

    for v, pred in zip(var_list, all_preds):
        pred = pred.squeeze().numpy()
        if testing_args.save_preds and v in testing_args.save_vars:
            if data_args.predict_steps == 1:
                torch.save(pred, os.path.join(preds_save_dir, f"{v}.pt"), pickle_protocol=4)
            else:
                for i in range(pred.shape[1]):
                    torch.save(pred[:, i], os.path.join(preds_save_dir, f"{v}_step{i+1}.pt"), pickle_protocol=4)

    with open(os.path.join(outdir, "init_times.json"), "w") as f:
        json.dump(predict_dataset.init_times, f, indent=4)

    save_prediction_metadata(
        outdir=outdir,
        init_times=predict_dataset.init_times,
        data_times=predict_dataset.times,
        input_steps=data_args.input_steps,
        predict_steps=data_args.predict_steps,
    )

    print("\n" + "*" * 10 + " Done " + "*" * 10)


if __name__ == "__main__":
    main()
