import os
from my_utils import get_logger,get_ckpt_filename,Vocab,DialogDataset,get_latest_ckpt,get_epoch_from_ckpt,f1_score
import torch
from torch.utils.data import DataLoader
import random
from model import Model
import traceback
from my_utils import PadBatchSeq
from tqdm import tqdm
config = {
  "vocab_path": "chinese_gpt_original/dict.txt",
  "max_seq_len": 45,
  "beam_size": 4,
  "diversity_coef": 0,
  "diversity_groups": 2,
  "sample": False,
  "annealing_topk": 20,
  "temperature": 0.8,
  "annealing": 0,
  "length_penalty": 2.2,
  "n_layers": 12,
  "n_pos_embeddings": 512,
  "embeddings_size": 768,
  "n_heads": 12,
  "dropout": 0.1,
  "embed_dropout": 0.1,
  "attn_dropout": 0.1,
  "ff_dropout": 0.1,
  "eval_steps": 100,

  "comment": "below is the training parameters",
  "n_epochs": 17,
  "batch_size": 16,
  "batch_split": 4,
  "lr": 6.25e-5,
  "lr_warmup": 1000,
  "lm_weight": 0.02,
  "risk_weight": 0,
  "n_jobs": 0,
  "label_smoothing": 0.1,
  "clip_grad": 1.0,
  "seed": 42,
  "load_last": False,

  "train_dir": "train",
  "eval_dir": "eval",
  "data_dir": "data",
  "log_dir": "log",
  "best_dir": "best",
  "train_data": "LCCC-train-small.txt",
  "valid_data": "LCCC-valid-small.txt",
  "test_data": "LCCC-test.txt",

  "cgpt_parameters_dir": "chinese_gpt_original/Cgpt_model.bin"
}

train_dir = 'train'
data_dir = 'data'
log_dir = 'log'
logger = get_logger('main.log')
out_file = 'infer_out.txt'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


try:
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))

    device = torch.device("cuda", 0)

    vocab = Vocab(config['vocab_path'])
    test_dataset = DialogDataset([os.path.join(data_dir, config['test_data'])],
                                          vocab, logger, config['max_seq_len'] - 1)

    test_dataloader = DataLoader(test_dataset, pin_memory=True,
                                 batch_size=config['batch_size'], collate_fn=PadBatchSeq(vocab.pad_id))

    logger.info('Building models')
    model = Model(config, vocab).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    latest_ckpt = 'model-16.ckpt'
    logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
    weights = torch.load(os.path.join(train_dir, latest_ckpt), map_location=device)['model']
    weight_keys = list(weights.keys())
    for key in weight_keys:
        if key.startswith('transformer_module.module'):
            weights['transformer_module' + key[len('transformer_module.module'):]] = weights[key]
            weights.pop(key)

    model.load_state_dict(weights, strict=True)

    with torch.no_grad():
        model.eval()
        res = []
        with open(os.path.join(os.path.dirname(out_file), os.path.basename(out_file)), 'w') as f:

            ITER = tqdm(test_dataloader, dynamic_ncols=True, total=len(test_dataloader))


            for data in ITER:
                prediction = model.predict([data['post'].to(device)])
                bs = data['post'].shape[0]
                for i in range(bs):
                    post_str = data['post'][i].tolist()[1:]
                    post_str = vocab.ids2string(post_str[:post_str.index(vocab.eos_id)])
                    resp_str = data['resp'][i].tolist()[1:]
                    resp_str = vocab.ids2string(resp_str[:resp_str.index(vocab.eos_id)])
                    pred_str = vocab.ids2string(prediction[i])
                    print('{}\t{}\t{}\t{}'.format(data['style'][i], post_str, pred_str, resp_str), file=f)

except:
    logger.error(traceback.format_exc())