import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from copy import deepcopy

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, polyak_update, obs_as_tensor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

SelfPPO = TypeVar("SelfPPO", bound="PPO")

class CustomPPO(PPO):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        use_advantage: bool = True,
        tau: float = 0.005, #soft update for regulized model
        regul_update_interval: int = 1,
        kl_coef: float = 0.0,
        kl_coef_decay: float = 0.0001,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size = batch_size,
            n_epochs = n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range = clip_range,
            clip_range_vf = clip_range_vf,
            normalize_advantage = normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl = target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,            
        )
        self.use_advantage = use_advantage
        self.tau = tau
        self.regul_update_interval = regul_update_interval
        self.regul_policy = deepcopy(self.policy)
        self.regul_policy.set_training_mode(False)
        self.kl_coef = kl_coef
        self.kl_coef_decay = kl_coef_decay
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        old_entropy = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            FER_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                #use advantages or returns
                gradient_coefs = rollout_data.advantages if self.use_advantage else rollout_data.returns
                # Normalize advantage
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(gradient_coefs) > 1:
                    gradient_coefs = (gradient_coefs - gradient_coefs.mean()) / (gradient_coefs.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = gradient_coefs * ratio
                policy_loss_2 = gradient_coefs * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                if self.use_advantage:
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                #entropy_loss = -th.mean(-log_prob)

                entropy_losses.append(entropy_loss.item())
                
                old_entropy.append(-rollout_data.old_log_prob.detach().cpu().numpy())

                loss = policy_loss + self.ent_coef * entropy_loss
                if self.use_advantage:
                    loss += self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                   log_ratio = log_prob - rollout_data.old_log_prob
                   approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                   approx_kl_divs.append(approx_kl_div)

                _, regul_log_prob, _ = self.regul_policy.evaluate_actions(rollout_data.observations, actions)
                FER_log_ratio = log_prob - regul_log_prob
                FER_kl_div = th.mean((th.exp(FER_log_ratio) - 1) - FER_log_ratio)
                FER_kl_div_value = FER_kl_div.cpu().detach().numpy()
                FER_kl_divs.append(FER_kl_div_value)
                # log_ratio = log_prob - rollout_data.old_log_prob.detach()
                # approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                # approx_kl_div = -th.mean(th.exp(rollout_data.old_log_prob.detach())*log_ratio)
                # approx_kl_divs.append(approx_kl_div.detach().cpu().numpy())
                
                #Cut the kl_div if larger than kl_target
                if FER_kl_div is not None and FER_kl_div_value < 2 * self.target_kl:
                    # loss -= self.kl_coef * (self.target_kl - abs(self.target_kl - FER_kl_div))
                    loss += self.kl_coef * FER_kl_div
                else:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max FER_kl: {FER_kl_div_value:.2f}")
                    break
                self.kl_coef = self.kl_coef * (1 - self.kl_coef_decay)
                
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
            # Update regul net
            if epoch % self.regul_update_interval == 0:
                polyak_update(self.policy.parameters(), self.regul_policy.parameters(), self.tau)
                
            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/old_entropy", np.mean(old_entropy))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        if self.use_advantage:
            self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/FER_kl", np.mean(FER_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/kl_coef", self.kl_coef)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            
    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     rollout_buffer: RolloutBuffer,
    #     n_rollout_steps: int,
    # ) -> bool:
    #     """
    #     Getting rollouts from regulization policy.
    #     """
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.regul_policy.set_training_mode(False)

    #     n_steps = 0
    #     rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.regul_policy.reset_noise(env.num_envs)

    #     callback.on_rollout_start()

    #     while n_steps < n_rollout_steps:
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.pregul_olicy.reset_noise(env.num_envs)

    #         with th.no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)
    #             actions, values, log_probs = self.regul_policy(obs_tensor)
    #         actions = actions.cpu().numpy()

    #         # Rescale and perform action
    #         clipped_actions = actions

    #         if isinstance(self.action_space, spaces.Box):
    #             if self.regul_policy.squash_output:
    #                 # Unscale the actions to match env bounds
    #                 # if they were previously squashed (scaled in [-1, 1])
    #                 clipped_actions = self.regul_policy.unscale_action(clipped_actions)
    #             else:
    #                 # Otherwise, clip the actions to avoid out of bound error
    #                 # as we are sampling from an unbounded Gaussian distribution
    #                 clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         new_obs, rewards, dones, infos = env.step(clipped_actions)

    #         self.num_timesteps += env.num_envs

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         if callback.on_step() is False:
    #             return False

    #         self._update_info_buffer(infos)
    #         n_steps += 1

    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)

    #         # Handle timeout by bootstraping with value function
    #         # see GitHub issue #633
    #         for idx, done in enumerate(dones):
    #             if (
    #                 done
    #                 and infos[idx].get("terminal_observation") is not None
    #                 and infos[idx].get("TimeLimit.truncated", False)
    #             ):
    #                 terminal_obs = self.regul_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #                 with th.no_grad():
    #                     terminal_value = self.regul_policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
    #                 rewards[idx] += self.gamma * terminal_value

    #         rollout_buffer.add(
    #             self._last_obs,  # type: ignore[arg-type]
    #             actions,
    #             rewards,
    #             self._last_episode_starts,  # type: ignore[arg-type]
    #             values,
    #             log_probs,
    #         )
    #         self._last_obs = new_obs  # type: ignore[assignment]
    #         self._last_episode_starts = dones

    #     with th.no_grad():
    #         # Compute value for the last timestep
    #         values = self.regul_policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

    #     rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    #     callback.update_locals(locals())

    #     callback.on_rollout_end()

    #     return True

