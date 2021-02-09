import torch
import os
import random
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import math
import torch.tensor
from my_utils import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, config, log_dir, logger, device=torch.device('cuda'),
                 ignore_idxs=[]):
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.valid_dataset = valid_dataset
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=60)
        self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        self.ignore_idxs = ignore_idxs
        self.model = model.to(device)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.vocab.pad_id).to(device)
        self.criterion = LabelSmoothingLoss(n_labels=len(self.model.vocab), smoothing=config['label_smoothing'],
                                            ignore_index=self.model.vocab.pad_id).to(device)
        base_optimizer = Adam(self.model.parameters(), lr=config['lr'], weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config['embeddings_size'], 0.1, config['lr_warmup'], base_optimizer)


        self.train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                           num_workers=config['n_jobs'], pin_memory=True,
                                           collate_fn=PadBatchSeq(self.model.vocab.pad_id))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                                           num_workers=config['n_jobs'], pin_memory=True,
                                           collate_fn=PadBatchSeq(self.model.vocab.pad_id))

    def perplexity(outputs, targets, config=None):
        """
        计算语言模型困惑度
        :param outputs: [batch_size,seq_len,vocab_size]
        :param targets: [batch_size,seq_len]
        :param config:  配置文件 default:None
        :return: 困惑度数值
        """
        ce = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1),
                             ignore_index=config.data.pad_id if config is not None else None)

        return torch.exp(ce)


    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch):
        self.model.train()
        loss = 0
        lm_loss = 0
        log_lm_loss, log_s2s_loss, step_count = 0, 0, 0
        total = len(self.train_dataloader)

        ITER = tqdm(enumerate(self.train_dataloader), dynamic_ncols=True, total=total)


        for i, data in ITER:
            post, resp = data['post'].to(self.device), data['resp'].to(self.device)
            enc_contexts = []

            # lm loss
            post_rep = self.model.encode(post.clone())
            enc_contexts.append(post_rep)

            post_outputs = self.model.generate(post_rep[0])
            ignore_mask = torch.stack([post == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1).bool()
            post.masked_fill_(ignore_mask, self.model.vocab.pad_id)
            prevs, nexts = post_outputs[:, :-1, :].contiguous(), post[:, 1:].contiguous()
            batch_lm_loss = self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))

            # s2s loss
            prevs, nexts = resp[:, :-1].contiguous(), resp[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # optimization
            full_loss = (batch_lm_loss * self.config['lm_weight'] + batch_loss) / self.config['batch_split']
            full_loss.backward()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)

            log_lm_loss += batch_lm_loss.item()
            log_s2s_loss += batch_loss.item()
            step_count += 1

            if (i + 1) % self.config['batch_split'] == 0:
                if self.config['clip_grad'] is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.config['clip_grad'])
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # shit log if you are node 0 in every step

                log_lm_loss /= step_count
                log_s2s_loss /= step_count
                self.train_writer.add_scalar('loss/lm_loss', log_lm_loss, self.optimizer.curr_step())
                self.train_writer.add_scalar('loss/s2s_loss', log_s2s_loss, self.optimizer.curr_step())
                self.train_writer.add_scalar('ppl/s2s_loss', math.exp(log_s2s_loss), self.optimizer.curr_step())
                self.train_writer.add_scalar('loss/total_loss', log_lm_loss + log_s2s_loss,
                                             self.optimizer.curr_step())
                self.train_writer.add_scalar('lr/lr', self.optimizer.rate(), self.optimizer.curr_step())
                log_lm_loss, log_s2s_loss, step_count = 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if self.optimizer.curr_step() % self.config['eval_steps'] == 0:
                    valid_lm_loss, valid_s2s_loss,perplexity = self._eval_test()
                    valid_lm_loss = valid_lm_loss.item()
                    valid_s2s_loss = valid_s2s_loss.item()
                    self.valid_writer.add_scalar('loss/lm_loss', valid_lm_loss, self.optimizer.curr_step())
                    self.valid_writer.add_scalar('loss/s2s_loss', valid_s2s_loss, self.optimizer.curr_step())
                    self.valid_writer.add_scalar('ppl/s2s_loss', math.exp(valid_s2s_loss), self.optimizer.curr_step())
                    self.valid_writer.add_scalar(
                        'loss/total_loss', valid_s2s_loss + valid_lm_loss, self.optimizer.curr_step())

                    log_str = ('epoch {:>3}, t_lm_loss {:>4.4f}, t_s2s_loss {:>4.4f}, ' +
                               'v_lm_loss {:>4.4f}, v_s2s_loss {:>4.4f}, perplexity {:>4.4f} lr {:>.6}, step {}').format(
                        epoch, lm_loss, loss, valid_lm_loss, valid_s2s_loss, perplexity,self.optimizer.rate(),
                        self.optimizer.curr_step())
                    self.logger.info(log_str)

                    # and only predicts sample on node 0
                    sample_dialog = self._pred_sample(5)
                    for j, d in enumerate(sample_dialog):
                        self.logger.info('--epoch {} step{} sample {}--'.format(
                            epoch, self.optimizer.curr_step(), j))
                        self.logger.info('post: {}'.format(d['post']))
                        self.logger.info('resp: {}'.format(d['resp']))
                        self.logger.info('pred: {}'.format(d['pred']))
                        self.train_writer.add_text('dialog', 'Post: {}\n  Resp: {}\n  Pred: {}\n'.format(
                            d['post'], d['resp'], d['pred']), self.optimizer.curr_step())
                    self.model.train()

    def _eval_test(self):
        loss = torch.tensor(0, dtype=torch.long, device=self.device)
        lm_loss = torch.tensor(0, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.model.eval()
            # self.logger.info("evaluating on rank {}, with datasize {}".format(self.rank, len(self.valid_dataloader)))
            for i, data in enumerate(self.valid_dataloader):
                post, resp = data['post'].to(self.device), data['resp'].to(self.device)
                enc_contexts = []

                # lm loss
                post_rep = self.model.encode(post.clone())
                enc_contexts.append(post_rep)

                context_outputs = self.model.generate(post_rep[0])
                ignore_mask = torch.stack([post == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1).bool()
                post.masked_fill_(ignore_mask, self.model.vocab.pad_id)
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), post[:, 1:].contiguous()
                batch_lm_loss = self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))

                # s2s loss
                prevs, nexts = resp[:, :-1].contiguous(), resp[:, 1:].contiguous()
                outputs = self.model.decode(prevs, enc_contexts)
                outputs = F.log_softmax(outputs, dim=-1)
                batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
                perplexity = self.perplexity(outputs.view(-1, outputs.shape[-1]), nexts.view(-1),)
                # predictions = self.model.beam_search(enc_contexts)
                # target_lens = resp.ne(self.model.padding_idx).sum(dim=-1)
                # targets = [t[1:l - 1].tolist() for t, l in zip(resp, target_lens)]

                lm_loss = (i * lm_loss + batch_lm_loss) / (i + 1)
                loss = (i * loss + batch_loss) / (i + 1)
            # self.logger.info("results on rank {}, {}, {}".format(self.rank, loss.item(), lm_loss.item()))
        # log_str = 'lm_loss {}, loss {}'.format(lm_loss, loss)
        # self.logger.info(log_str)
        return lm_loss, loss,perplexity

    def _pred_sample(self, n_sample):
        with torch.no_grad():
            self.model.eval()
            samples_idxs = random.sample(range(len(self.valid_dataset)), n_sample)
            samples = PadBatchSeq(self.model.vocab.pad_id)([self.valid_dataset[idx] for idx in samples_idxs])
            prediction = self.model.predict([samples['post'].to(self.device)])
            res = []
            for j in range(len(samples_idxs)):
                post_str = samples['post'][j].tolist()[1:]
                post_str = self.model.vocab.ids2string(post_str[:post_str.index(self.model.vocab.eos_id)])
                resp_str = samples['resp'][j].tolist()[1:]
                resp_str = self.model.vocab.ids2string(resp_str[:resp_str.index(self.model.vocab.eos_id)])
                pred_str = self.model.vocab.ids2string(prediction[j])
                res.append({"post": post_str, "resp": resp_str, "pred": pred_str})

        return res

    def test(self):
        self._eval_test()

    def train(self, start_epoch, epochs, after_epoch_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on epoch {}, step {}'.format(
             epoch, self.optimizer.curr_step()))
            self._eval_train(epoch)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch, self.device)

#loss function
class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='batchmean')
            n_ignore_idxs = 1 + (ignore_index >= 0)  # 1 for golden truth, later one for ignore_index
            one_hot = torch.full((1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs)))
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            self.criterion = nn.NLLLoss(reduction='mean', ignore_index=ignore_index)

    def forward(self, log_inputs, targets):
        if self.confidence < 1:
            tdata = targets.data

            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp

        return self.criterion(log_inputs, targets)

#optimizer
class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.
    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class NoamOpt:
    def __init__(self, embeddings_size, factor, warmup, optimizer):
        self.embeddings_size = embeddings_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer

        self._step = 1

    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def curr_step(self):
        return self._step

    def rate(self, step=None):
        if step is None:
            step = self._step

        return self.factor * (self.embeddings_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))