from torch.nn import Module
from torch.nn.functional import log_softmax, nll_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Sequence, Tuple
from time import time
import math
import torch
from operator import lt, gt
from numpy.random import Generator
from uncertainty import count_correct, logmeanexp, nll_loss_from_probs
from uncertainty import (
    epig_from_logprobs,
    epig_from_logprobs_using_matmul,
    epig_from_logprobs_using_weights,
    epig_from_probs,
    epig_from_probs_using_matmul,
    epig_from_probs_using_weights,
)
from utils import Dictionary
from torch import Tensor
from torch.nn.functional import nll_loss
from tqdm import tqdm
from typing import List, Tuple

class Trainer : 
    def __init__(self) :
        pass
    def eval_mode(self) :
        pass
    def conditional_predict(self, input : Tensor, n_model_samples : int, independent : bool) :
        pass

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> None:
        pass

    def evaluate(self, inputs: Tensor, labels: Tensor, n_model_samples: int) -> None:
        pass

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        pass

    def test(self, loader: DataLoader) -> Tuple[Tensor, Tensor]:
        self.eval_mode()
        total_correct = total_loss = n_examples = 0
        for inputs, labels in loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            n_correct, loss = self.evaluate(inputs, labels, self.n_samples_test)  # [1,], [1,]
            total_correct += n_correct  # [1,]
            total_loss += loss * len(inputs)  # [1,]
            n_examples += len(inputs)  # [1,]

        acc = total_correct / n_examples  # [1,]
        loss = total_loss / n_examples  # [1,]
        return acc, loss  # [1,], [1,]

    def estimate_uncertainty(
        self,
        pool_loader: DataLoader,
        target_inputs: Tensor,
        mode: str,
        rng: Generator,
        epig_probs_target: List[float] = None,
        epig_probs_adjustment: List[float] = None,
        epig_using_matmul: bool = False,
    ) -> Dictionary:
        pool_loader = tqdm(pool_loader, desc="Uncertainty") if self.verbose else pool_loader

        if mode == "epig":
            if epig_probs_target != None:
                scores = self.estimate_epig_using_pool(
                    pool_loader, epig_probs_target, epig_probs_adjustment, len(target_inputs)
                )
            else:
                scores = self.estimate_epig(pool_loader, target_inputs, epig_using_matmul)

        elif mode == "random":
            scores = self.sample_uniform(pool_loader, rng)

        return scores

    def estimate_epig(
        self, loader: DataLoader, target_inputs: Tensor, use_matmul: bool
    ) -> Dictionary:
        self.eval_mode()
        scores = Dictionary()

        for inputs, _, _ in loader:
            epig_scores = self.estimate_epig_minibatch(inputs, target_inputs, use_matmul)  # [B,]
            scores.append({"epig": epig_scores.cpu()})

        return scores.concatenate()

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> None:
        pass

    def estimate_epig_using_pool(
        self,
        loader: DataLoader,
        probs_target: List[float],
        probs_adjustment: List[float],
        n_input_samples: int,
    ) -> None:
        pass

    def sample_uniform(self, loader: DataLoader, rng: Generator) -> Dictionary:
        n_inputs = len(loader.dataset.indices)
        samples = rng.uniform(size=n_inputs)
        samples = torch.tensor(samples)
        scores = Dictionary()
        scores.append({"random": samples})
        return scores.concatenate()


class LogProbsTrainer(Trainer):
    """
    Base trainer for a model that outputs log probabilities.
    """

    def evaluate(
        self, inputs: Tensor, labels: Tensor, n_model_samples: int
    ) -> Tuple[Tensor, Tensor]:
        logprobs = self.marginal_predict(inputs, n_model_samples)  # [N, Cl]
        n_correct = count_correct(logprobs, labels)  # [1,]
        loss = nll_loss(logprobs.cuda(), labels.type(torch.LongTensor).cuda())  # [1,]
        return n_correct, loss

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> Tensor:
        inputs = inputs.cuda()
        target_inputs = target_inputs.cuda()
        combined_inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]
        logprobs = self.conditional_predict(
            combined_inputs, self.n_samples_test, independent=False
        )  # [N + N_t, K, Cl]
        epig_fn = epig_from_logprobs_using_matmul if use_matmul else epig_from_logprobs
        return epig_fn(logprobs[: len(inputs)], logprobs[len(inputs) :])  # [N,]

    @torch.inference_mode()
    def estimate_epig_using_pool(
        self,
        loader: DataLoader,
        probs_target: List[float],
        probs_adjustment: List[float] = None,
        n_input_samples: int = None,
    ) -> Dictionary:
        self.eval_mode()

        logprobs_cond = []
        for inputs, _ in loader:
            logprobs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            logprobs_cond.append(logprobs_cond_i)
        logprobs_cond = torch.cat(logprobs_cond)  # [N, K, Cl]

        logprobs_marg = logmeanexp(logprobs_cond, dim=1)  # [N, Cl]
        logprobs_marg_marg = logmeanexp(logprobs_marg, dim=0, keepdim=True)  # [1, Cl]

        if probs_adjustment != None:
            probs_adjustment = torch.tensor([probs_adjustment])  # [1, Cl]
            probs_adjustment = probs_adjustment.to(inputs.device)  # [1, Cl]
            probs_marg = torch.exp(logprobs_marg)  # [N, Cl]
            probs_marg += probs_adjustment * torch.exp(logprobs_marg_marg)  # [N, Cl]
            probs_marg /= torch.sum(probs_marg, dim=-1, keepdim=True)  # [N, Cl]
            logprobs_marg = torch.log(probs_marg)  # [N, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        probs_target = torch.tensor([probs_target])  # [1, Cl]
        probs_target = probs_target.to(inputs.device)  # [1, Cl]
        log_ratio = logprobs_marg - logprobs_marg_marg  # [N, Cl]
        weights = torch.sum(probs_target * torch.exp(log_ratio), dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = Dictionary()

        if n_input_samples != None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            logprobs_target = logprobs_cond[inds]  # [N_s, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                epig_scores = epig_from_logprobs(logprobs_cond_i, logprobs_target)  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        else:
            logprobs_target = logprobs_cond  # [N, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                epig_scores = epig_from_logprobs_using_weights(
                    logprobs_cond_i, logprobs_target, weights
                )  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        return scores.concatenate()


class ProbsTrainer(Trainer):
    """
    Base trainer for a model that outputs probabilities.
    """

    def evaluate(
        self, inputs: Tensor, labels: Tensor, n_model_samples: int
    ) -> Tuple[Tensor, Tensor]:
        probs = self.marginal_predict(inputs, n_model_samples)  # [N, Cl]
        n_correct = count_correct(probs, labels)  # [1,]
        loss = nll_loss_from_probs(probs, labels)  # [1,]
        return n_correct, loss

    def estimate_epig_minibatch(
        self, inputs: Tensor, target_inputs: Tensor, use_matmul: bool
    ) -> Tensor:
        _inputs = torch.cat((inputs, target_inputs))  # [N + N_t, ...]
        probs = self.conditional_predict(
            _inputs, self.n_samples_test, independent=False
        )  # [N + N_t, K, Cl]
        epig_fn = epig_from_probs_using_matmul if use_matmul else epig_from_probs
        return epig_fn(probs[: len(inputs)], probs[len(inputs) :])  # [N,]

    @torch.inference_mode()
    def estimate_epig_using_pool(
        self,
        loader: DataLoader,
        probs_target: List[float],
        probs_adjustment: List[float] = None,
        n_input_samples: int = None,
    ) -> Dictionary:
        self.eval_mode()

        probs_cond = []
        for inputs, _ in loader:
            probs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            probs_cond.append(probs_cond_i)
        probs_cond = torch.cat(probs_cond)  # [N, K, Cl]

        probs_marg = torch.mean(probs_cond, dim=1)  # [N, Cl]
        probs_marg_marg = torch.mean(probs_marg, dim=0, keepdim=True)  # [1, Cl]

        if probs_adjustment != None:
            probs_adjustment = torch.tensor([probs_adjustment])  # [1, Cl]
            probs_adjustment = probs_adjustment.to(inputs.device)  # [1, Cl]
            probs_marg += probs_adjustment * probs_marg_marg  # [N, Cl]
            probs_marg /= torch.sum(probs_marg, dim=-1, keepdim=True)  # [N, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        probs_target = torch.tensor([probs_target])  # [1, Cl]
        probs_target = probs_target.to(inputs.device)  # [1, Cl]
        weights = torch.sum(probs_target * probs_marg / probs_marg_marg, dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = Dictionary()

        if n_input_samples != None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            probs_target = probs_cond[inds]  # [N_s, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                epig_scores = epig_from_probs(probs_cond_i, probs_target)  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        else:
            probs_target = probs_cond  # [N, K, Cl]

            for probs_cond_i in torch.split(probs_cond, len(inputs)):
                epig_scores = epig_from_probs_using_weights(
                    probs_cond_i, probs_target, weights
                )  # [B,]
                scores.append({"epig": epig_scores.cpu()})

        return scores.concatenate()

class NeuralNetworkTrainer(LogProbsTrainer):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        n_optim_steps_min: int,
        n_optim_steps_max: int,
        n_samples_train: int,
        n_samples_test: int,
        n_validations: int,
        early_stopping_metric: str,
        early_stopping_patience: int,
        restore_best_model: bool,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.n_optim_steps_min = n_optim_steps_min
        self.n_optim_steps_max = n_optim_steps_max
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.validation_gap = max(1, int(n_optim_steps_max / n_validations))
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_model = restore_best_model
        self.verbose = verbose

    def eval_mode(self) -> None:
        self.model.eval()

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        features = self.model(inputs, n_model_samples)  # [N, K, Cl]
        return log_softmax(features, dim=-1)  # [N, K, Cl]

    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        logprobs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]

        if n_model_samples == 1:
            return torch.squeeze(logprobs, dim=1)  # [N, Cl]
        else:
            return logmeanexp(logprobs, dim=1)  # [N, Cl]

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dictionary:
        log = Dictionary()
        start_time = time()

        step_range = range(self.n_optim_steps_max)
        step_range = tqdm(step_range, desc="Training") if self.verbose else step_range

        best_score = 0 if "acc" in self.early_stopping_metric else math.inf
        early_stopping_operator = gt if "acc" in self.early_stopping_metric else lt
        
        for step in step_range:
            train_acc, train_loss = self.train_step(train_loader)

            if step % self.validation_gap == 0:
                with torch.inference_mode():
                    val_acc, val_loss = self.test(val_loader)

                log_update = {
                    "time": time() - start_time,
                    "step": step,
                    "train_acc": train_acc.item(),
                    "train_loss": train_loss.item(),
                    "val_acc": val_acc.item(),
                    "val_loss": val_loss.item(),
                }
                log.append(log_update)

                print("step {}/{} | time {} | train_acc {} | train_loss {} | val_acc {} | val_loss {}".format(
                    step, self.n_optim_steps_max,time()-start_time,train_acc.item(),train_loss.item(),val_acc.item(),val_loss.item()))
                latest_score = log_update[self.early_stopping_metric]
                score_has_improved = early_stopping_operator(latest_score, best_score)

                if (step < self.n_optim_steps_min) or score_has_improved:
                    best_model_state = self.model.state_dict()
                    best_score = latest_score
                    patience_left = self.early_stopping_patience
                else:
                    patience_left -= self.validation_gap

                if (self.early_stopping_patience != -1) and (patience_left <= 0):
                   #logging.info(f"Stopping training at step {step}")
                   break

        if self.restore_best_model:
            self.model.load_state_dict(best_model_state)

        return log

    def train_step(self, loader: DataLoader) -> Tuple[Tensor, Tensor]:
        try:
            inputs, labels = next(iter(loader))
        except:
            loader = iter(loader)
            inputs, labels = next(iter(loader))

        inputs = inputs.cuda()[0]
        labels = labels.cuda()[0]

        self.model.train()
        self.optimizer.zero_grad()

        n_correct, loss = self.evaluate(inputs, labels, self.n_samples_train)  # [1,], [1,]
        acc = n_correct / len(inputs)  # [1,]

        loss.backward()
        self.optimizer.step()

        return acc, loss  # [1,], [1,]

    def compute_badge_embeddings(
        self, loader: DataLoader, embedding_params: Sequence[str]
    ) -> Tensor:
        self.eval_mode()

        embeddings = []

        for inputs, _ in loader:
            pseudolosses = self.compute_pseudoloss(inputs)  # [B,]

            for pseudoloss in pseudolosses:
                # Prevent the grad attribute of each tensor accumulating a sum of gradients.
                self.model.zero_grad()

                pseudoloss.backward(retain_graph=True)

                gradients = []

                for name, param in self.model.named_parameters():
                    if name in embedding_params:
                        gradient = param.grad.flatten().cpu()  # [E_i,]
                        gradients.append(gradient)

                embedding = torch.cat(gradients)  # [E,]
                embeddings.append(embedding)

        return torch.stack(embeddings)  # [N, E]

    def compute_pseudoloss(self, inputs: Tensor) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, ...]

        Returns:
            Tensor[float], [N,]
        """
        logprobs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]
        pseudolabels = torch.argmax(logprobs, dim=-1)  # [N,]
        return nll_loss(logprobs, pseudolabels, reduction="none")  # [N,]