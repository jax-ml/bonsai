# Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: check https://huggingface.co/papers/2302.04867 and https://github.com/wl-zhao/UniPC

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from .scheduling_utils import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
)


@flax.struct.dataclass
class UniPCMultistepSchedulerState:
    common: CommonSchedulerState
    alpha_t: jnp.ndarray
    sigma_t: jnp.ndarray
    lambda_t: jnp.ndarray

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    sigmas: jnp.ndarray
    num_inference_steps: Optional[int] = None

    # running values
    model_outputs: Optional[jnp.ndarray] = None
    timestep_list: Optional[jnp.ndarray] = None
    lower_order_nums: Optional[int] = None
    this_order: Optional[int] = None
    last_sample: Optional[jnp.ndarray] = None  # For UniC corrector

    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,
        alpha_t: jnp.ndarray,
        sigma_t: jnp.ndarray,
        lambda_t: jnp.ndarray,
        init_noise_sigma: jnp.ndarray,
        timesteps: jnp.ndarray,
        sigmas: jnp.ndarray,
    ):
        return cls(
            common=common,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            lambda_t=lambda_t,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
        )


@dataclass
class FlaxUniPCMultistepSchedulerOutput(FlaxSchedulerOutput):
    state: UniPCMultistepSchedulerState


class FlaxUniPCMultistepScheduler(FlaxSchedulerMixin):
    """
    `FlaxUniPCMultistepScheduler` is a JAX/Flax port of the UniPC multistep scheduler.

    This scheduler is designed for fast sampling of diffusion models with flow_shift support for Wan models.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule. Choose from `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        solver_order (`int`, defaults to 2):
            The UniPC order. Recommended to use 2 for guided sampling, 3 for unconditional.
        prediction_type (`str`, defaults to `"epsilon"`):
            Prediction type: `epsilon`, `sample`, or `v_prediction`.
        use_flow_sigmas (`bool`, defaults to False):
            Whether to use flow sigmas with flow_shift transformation.
        flow_shift (`float`, defaults to 1.0):
            Flow shift parameter for timestep transformation. Use 5.0 for 720P, 3.0 for 480P.
        timestep_spacing (`str`, defaults to `"linspace"`):
            How to scale timesteps. Choose from `linspace`, `leading`, or `trailing`.
        predict_x0 (`bool`, defaults to True):
            Whether to use the updating algorithm on the predicted x0.
        solver_type (`str`, defaults to `"bh2"`):
            Solver type. Use `bh1` for <10 steps, `bh2` otherwise.
        dtype (`jnp.dtype`, defaults to `jnp.float32`):
            The dtype used for params and computation.
    """

    dtype: jnp.dtype

    @property
    def has_state(self):
        return True

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        use_flow_sigmas: bool = False,
        flow_shift: float = 1.0,
        timestep_spacing: str = "linspace",
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

    def create_state(self, common: Optional[CommonSchedulerState] = None) -> UniPCMultistepSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)

        # VP-type noise schedule
        alpha_t = jnp.sqrt(common.alphas_cumprod)
        sigma_t = jnp.sqrt(1 - common.alphas_cumprod)
        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)

        # Sigmas for noise schedule
        sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5

        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        return UniPCMultistepSchedulerState.create(
            common=common,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            lambda_t=lambda_t,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
        )

    def set_timesteps(
        self,
        state: UniPCMultistepSchedulerState,
        num_inference_steps: int,
        shape: Tuple,
    ) -> UniPCMultistepSchedulerState:
        """
        Sets the discrete timesteps used for the diffusion chain.

        Args:
            state (`UniPCMultistepSchedulerState`):
                The scheduler state.
            num_inference_steps (`int`):
                The number of diffusion steps.
            shape (`Tuple`):
                The shape of the samples to be generated.
        """
        # Timestep generation
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                jnp.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .astype(jnp.int32)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // (num_inference_steps + 1)
            timesteps = (jnp.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].astype(jnp.int32)
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = jnp.arange(self.config.num_train_timesteps, 0, -step_ratio).round().astype(jnp.int32)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Choose from 'linspace', 'leading', or 'trailing'."
            )

        # Sigma calculation with flow_shift support
        if self.config.use_flow_sigmas:
            # Flow-based sigma schedule (for Wan models)
            alphas = jnp.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
            sigmas_raw = 1.0 - alphas
            # Apply flow_shift transformation
            flow_shift = self.config.flow_shift
            sigmas = jnp.flip(flow_shift * sigmas_raw / (1 + (flow_shift - 1) * sigmas_raw))[:-1]
            timesteps = (sigmas * self.config.num_train_timesteps).astype(jnp.int32)
        else:
            # Standard sigma schedule
            sigmas = jnp.array(((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod) ** 0.5)
            sigmas = sigmas[timesteps]

        # Initialize running values
        model_outputs = jnp.zeros((self.config.solver_order, *shape), dtype=self.dtype)
        timestep_list = jnp.zeros((self.config.solver_order,), dtype=jnp.int32)
        lower_order_nums = jnp.int32(0)
        this_order = jnp.int32(self.config.solver_order)

        return state.replace(
            timesteps=timesteps,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            model_outputs=model_outputs,
            timestep_list=timestep_list,
            lower_order_nums=lower_order_nums,
            this_order=this_order,
        )

    def convert_model_output(
        self,
        state: UniPCMultistepSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        step_index: int,
    ) -> jnp.ndarray:
        """
        Convert the model output to the corresponding type the UniPC algorithm needs.
        """
        # Get sigma for current step
        sigma = state.sigmas[step_index]

        # sigma_to_alpha_sigma_t conversion
        alpha_t = 1.0 / jnp.sqrt(1 + sigma**2)
        sigma_t = sigma * alpha_t

        if self.config.predict_x0:
            if self.config.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type {self.config.prediction_type} not supported. "
                    "Choose from 'epsilon', 'sample', or 'v_prediction'."
                )
            return x0_pred
        else:
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(f"prediction_type {self.config.prediction_type} not supported.")

    def multistep_uni_c_bh_update(
        self,
        state: UniPCMultistepSchedulerState,
        this_model_output: jnp.ndarray,
        last_sample: jnp.ndarray,
        this_sample: jnp.ndarray,
        order: int,
        step_index: int,
    ) -> jnp.ndarray:
        """
        One step for the UniC (B(h) version) corrector.

        Args:
            state: The scheduler state
            this_model_output: The model outputs at current timestep x_t
            last_sample: The generated sample before last predictor x_{t-1}
            this_sample: The generated sample after last predictor x_{t}
            order: The order of UniC at this step
            step_index: Current step index

        Returns:
            The corrected sample tensor at the current timestep.
        """
        model_output_list = state.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t = state.sigmas[step_index]
        sigma_s0 = state.sigmas[step_index - 1]

        # sigma_to_alpha_sigma_t conversion
        alpha_t = 1.0 / jnp.sqrt(1 + sigma_t**2)
        sigma_t_converted = sigma_t * alpha_t
        alpha_s0 = 1.0 / jnp.sqrt(1 + sigma_s0**2)
        sigma_s0_converted = sigma_s0 * alpha_s0

        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t_converted)
        lambda_s0 = jnp.log(alpha_s0) - jnp.log(sigma_s0_converted)

        h = lambda_t - lambda_s0

        # Build rks and D1s for higher order
        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            sigma_si = state.sigmas[si]
            alpha_si = 1.0 / jnp.sqrt(1 + sigma_si**2)
            sigma_si_converted = sigma_si * alpha_si
            lambda_si = jnp.log(alpha_si) - jnp.log(sigma_si_converted)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = jnp.array(rks)

        # Build R matrix and b vector
        hh = -h if self.config.predict_x0 else h
        h_phi_1 = jnp.expm1(hh)  # e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = jnp.expm1(hh)
        else:
            raise NotImplementedError(f"solver_type {self.config.solver_type} not implemented")

        R = []
        b = []
        factorial_i = 1

        for i in range(1, order + 1):
            R.append(rks ** (i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = jnp.stack(R)
        b = jnp.array(b)

        # Solve for correction coefficients
        if order == 1:
            rhos_c = jnp.array([0.5])
        else:
            rhos_c = jnp.linalg.solve(R, b)

        # Apply correction
        if self.config.predict_x0:
            x_t_ = sigma_t_converted / sigma_s0_converted * x - alpha_t * h_phi_1 * m0

            if len(D1s) > 0:
                D1s_stack = jnp.stack(D1s, axis=0)  # (order-1, *sample_shape)
                # Compute correction residual: sum of rhos_c[:-1] * D1s
                corr_res = jnp.tensordot(rhos_c[:-1], D1s_stack, axes=[[0], [0]])
            else:
                corr_res = 0

            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t_converted * h_phi_1 * m0

            if len(D1s) > 0:
                D1s_stack = jnp.stack(D1s, axis=0)
                corr_res = jnp.tensordot(rhos_c[:-1], D1s_stack, axes=[[0], [0]])
            else:
                corr_res = 0

            D1_t = model_t - m0
            x_t = x_t_ - sigma_t_converted * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t

    def multistep_uni_p_bh_update(
        self,
        state: UniPCMultistepSchedulerState,
        model_output: jnp.ndarray,
        sample: jnp.ndarray,
        order: int,
        step_index: int,
    ) -> jnp.ndarray:
        """
        One step for the UniP (B(h) version) predictor.
        """
        timestep = state.timesteps[step_index]
        prev_timestep = jax.lax.select(
            step_index == len(state.timesteps) - 1, jnp.int32(0), state.timesteps[step_index + 1]
        )

        model_output_list = [state.model_outputs[-(i + 1)] for i in range(order)]

        # Get lambda values
        lambda_t = state.lambda_t[timestep]
        lambda_s = state.lambda_t[prev_timestep]

        h = lambda_t - lambda_s

        # First order update
        def order_1_update():
            _sigma_t = state.sigma_t[timestep]
            sigma_s = state.sigma_t[prev_timestep]
            alpha_t = state.alpha_t[timestep]
            alpha_s = state.alpha_t[prev_timestep]

            x_t = sample
            if self.config.predict_x0:
                x0_t = model_output_list[0]
                x_s = (alpha_s / alpha_t) * x_t - (sigma_s * jnp.expm1(-h)) * x0_t
            else:
                epsilon_t = model_output_list[0]
                x_s = (alpha_s / alpha_t) * x_t - (sigma_s * jnp.expm1(-h)) * epsilon_t
            return x_s

        # Second order update
        def order_2_update():
            timestep_prev = state.timestep_list[-(order - 1)]

            lambda_s1 = state.lambda_t[timestep_prev]
            h_0 = lambda_t - lambda_s1

            r0 = h_0 / h

            _sigma_t = state.sigma_t[timestep]
            sigma_s = state.sigma_t[prev_timestep]
            alpha_t = state.alpha_t[timestep]
            alpha_s = state.alpha_t[prev_timestep]

            x_t = sample

            if self.config.predict_x0:
                if self.config.solver_type == "bh1":
                    x0_t = model_output_list[0]
                    x0_s1 = model_output_list[1]

                    D1 = (1.0 / r0) * (x0_t - x0_s1)
                    x_s = (
                        (alpha_s / alpha_t) * x_t
                        - (sigma_s * jnp.expm1(-h)) * x0_t
                        - 0.5 * (sigma_s * jnp.expm1(-h)) * D1
                    )
                elif self.config.solver_type == "bh2":
                    x0_t = model_output_list[0]
                    x0_s1 = model_output_list[1]

                    D1 = (1.0 / r0) * (x0_t - x0_s1)
                    corr_res = (1.0 / (2.0 * r0)) * D1
                    x_s = (alpha_s / alpha_t) * x_t - (sigma_s * jnp.expm1(-h)) * x0_t + (sigma_s / alpha_s) * corr_res
                else:
                    raise NotImplementedError(f"solver_type {self.config.solver_type} not implemented")
            else:
                # Epsilon prediction (not commonly used with UniPC)
                epsilon_t = model_output_list[0]
                epsilon_s1 = model_output_list[1]

                D1 = (1.0 / r0) * (epsilon_t - epsilon_s1)
                x_s = (
                    (alpha_s / alpha_t) * x_t
                    - (sigma_s * jnp.expm1(-h)) * epsilon_t
                    - 0.5 * (sigma_s * jnp.expm1(-h)) * D1
                )
            return x_s

        # Use conditional to select order
        return jax.lax.cond(
            order == 1,
            order_1_update,
            order_2_update,
        )

    def step(
        self,
        state: UniPCMultistepSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxUniPCMultistepSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by UniPC multistep.

        Args:
            state (`UniPCMultistepSchedulerState`):
                The scheduler state.
            model_output (`jnp.ndarray`):
                Direct output from learned diffusion model.
            timestep (`int`):
                Current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                Current instance of sample being created by diffusion process.
            return_dict (`bool`):
                Whether to return dict or tuple.

        Returns:
            [`FlaxUniPCMultistepSchedulerOutput`] or `tuple`:
                Updated sample and state.
        """
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # Find step index
        step_index = jnp.where(state.timesteps == timestep, size=1)[0][0]

        # Convert model output
        model_output_convert = self.convert_model_output(state, model_output, timestep, sample, step_index)

        # Apply UniC corrector (if not first step and last_sample exists)
        use_corrector = jnp.logical_and(step_index > 0, state.last_sample is not None)

        sample_corrected = jax.lax.cond(
            use_corrector,
            lambda: self.multistep_uni_c_bh_update(
                state,
                model_output_convert,
                state.last_sample,
                sample,
                jnp.minimum(state.this_order, state.lower_order_nums + 1),
                step_index,
            ),
            lambda: sample,  # No correction on first step
        )

        # Update model outputs buffer (rolling)
        model_outputs_new = jnp.roll(state.model_outputs, -1, axis=0)
        model_outputs_new = model_outputs_new.at[-1].set(model_output_convert)

        # Update timestep list buffer (rolling)
        timestep_list_new = jnp.roll(state.timestep_list, -1, axis=0)
        timestep_list_new = timestep_list_new.at[-1].set(timestep)

        # Determine order for this step
        this_order = jax.lax.select(
            self.config.lower_order_final,
            jnp.minimum(self.config.solver_order, len(state.timesteps) - step_index),
            jnp.int32(self.config.solver_order),
        )
        this_order = jnp.minimum(this_order, state.lower_order_nums + 1)

        # Apply UniP predictor (using corrected sample)
        prev_sample = self.multistep_uni_p_bh_update(
            state.replace(model_outputs=model_outputs_new, timestep_list=timestep_list_new),
            model_output,
            sample_corrected,  # Use corrected sample!
            this_order,
            step_index,
        )

        # Update state (store current sample for next corrector step)
        state = state.replace(
            model_outputs=model_outputs_new,
            timestep_list=timestep_list_new,
            lower_order_nums=jnp.minimum(state.lower_order_nums + 1, self.config.solver_order),
            this_order=this_order,
            last_sample=sample_corrected,  # Save for next corrector
        )

        if not return_dict:
            return (prev_sample, state)

        return FlaxUniPCMultistepSchedulerOutput(prev_sample=prev_sample, state=state)

    def add_noise(
        self,
        state: UniPCMultistepSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        sqrt_alpha_prod = state.common.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, original_samples)

        sqrt_one_minus_alpha_prod = (1 - state.common.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
