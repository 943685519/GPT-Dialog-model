import os
from my_utils import get_logger,Vocab,get_latest_ckpt
import torch
import random
from model import Model
import traceback
from trainer import Trainer

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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:

    model_path = os.path.join(train_dir, get_latest_ckpt(train_dir))

    if not os.path.isfile(model_path):
        print('cannot find {}'.format(model_path))
        exit(0)

    device = torch.device("cuda")

    vocab = Vocab(config['vocab_path'])

    print('Building models')
    model =  Model(config, vocab).to(device)

    print('Loading weights from {}'.format(model_path))
    state_dict = torch.load(model_path, map_location=device)['model']
    for i in list(state_dict.keys()):
        state_dict[i.replace('.module.', '.')] = state_dict.pop(i)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        post = input('>> ')
        post = ' '.join(list(post.replace(' ', '')))
        post = [vocab.eos_id] + vocab.string2ids(post) + [vocab.eos_id]
        contexts = [torch.tensor([post], dtype=torch.long, device=device)]
        prediction = model.predict(contexts)[0]
        pred_str = vocab.ids2string(prediction)
        print('小爱同学: {}'.format(pred_str))

except:
    print(traceback.format_exc())