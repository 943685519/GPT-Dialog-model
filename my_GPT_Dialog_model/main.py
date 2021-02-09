import os
from my_utils import get_logger,get_ckpt_filename,Vocab,DialogDataset,get_latest_ckpt,get_epoch_from_ckpt,f1_score
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


def save_func(epoch, device):
  filename = get_ckpt_filename('model', epoch)
  torch.save(trainer.state_dict(), os.path.join(train_dir, filename))
  if os.path.exists(os.path.join(train_dir, get_ckpt_filename('model', epoch - 80))):
    os.remove(os.path.join(train_dir, get_ckpt_filename('model', epoch - 80)))

def sample_text_func(epoch, device):
  n_samples = 8
  samples_idxs = random.sample(range(len(valid_dataset)), n_samples)
  samples = [valid_dataset[idx] for idx in samples_idxs]
  for i, data in enumerate(samples):
    contexts = [torch.tensor([data['post']], dtype=torch.long, device=device)]

    prediction = trainer.model.predict(contexts)[0]
    post_str = vocab.ids2string(data['post'][1:-1])
    resp_str = vocab.ids2string(data['resp'][1:-1])
    pred_str = vocab.ids2string(prediction)

    logger.info('-------epoch {} sample {}---------'.format(epoch, i))
    logger.info('post: {}'.format(post_str))
    logger.info('resp: {}'.format(resp_str))
    logger.info('pred: {}'.format(pred_str))


try:
  logger.info('pytorch version: {}'.format(torch.__version__))
  for i in config:
      logger.info('{}: {}'.format(i, config[i]))

  dirs = [train_dir, log_dir]

  for d in dirs:
      if not os.path.isdir(d):
          logger.info('cannot find {}, mkdiring'.format(d))
          os.makedirs(d)

  device = torch.device("cuda", 0)

  vocab = Vocab(config["vocab_path"])
  train_dataset = DialogDataset([os.path.join(data_dir, config['train_data'])],
                                        vocab, logger, config['max_seq_len'] - 1)
  valid_dataset = DialogDataset([os.path.join(data_dir, config['valid_data'])],
                                        vocab, logger, config['max_seq_len'] - 1)

  logger.info('Building models')
  model =  Model(config, vocab).to(device)
  for name, param in model.named_parameters():
      if param.requires_grad:
          print(name, param.shape)

  latest_ckpt = get_latest_ckpt(train_dir)
  if latest_ckpt is None:  # start from scratch
      logger.info('start from CGPT weights')
      cgpt_model = torch.load(config['cgpt_parameters_dir'], map_location=device)
      cgpt_model.pop('decoder.pre_softmax.weight')

      b = list(cgpt_model.keys())
      for i in b:
          cgpt_model[i.split('.', 1)[1]] = cgpt_model.pop(i)
      model.transformer_module.load_state_dict(cgpt_model, strict=True)
      logger.info('CGPT weights loaded from {}'.format(config['cgpt_parameters_dir']))

  trainer = Trainer(model, train_dataset, valid_dataset, config, log_dir, logger, device, vocab.special_tokens_ids,
                    )



  start_epoch = 0
  if latest_ckpt is not None:
      logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
      start_epoch = get_epoch_from_ckpt(latest_ckpt)
      trainer.load_state_dict(torch.load(os.path.join(train_dir, latest_ckpt), map_location=device))

  try:
      trainer.train(start_epoch, config['n_epochs'], after_epoch_funcs=[save_func])
      trainer.train(config['n_epochs'], after_epoch_funcs=[sample_text_func])
  except (KeyboardInterrupt, Exception, RuntimeError) as e:
      torch.save(trainer.state_dict(), os.path.join(train_dir, 'interrupt.pt'))
      raise e
except:
    logger.error(traceback.format_exc())