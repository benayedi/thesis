# modified_vae.py
import logging
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.model import SCVI

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial, Poisson

from scvi import REGISTRY_KEYS


torch.backends.cudnn.benchmark = True

llogger = logging.getLogger(__name__)

def quasi_likelihood_loss(reconstructed_mean, target, dispersion, power):
    if not isinstance(power, torch.Tensor):
        power = torch.tensor(power)
    if power.numel() != 1:
        raise ValueError("power must be a scalar or a single element tensor")
    residual = torch.abs(target - reconstructed_mean)
    variance = dispersion * torch.pow(reconstructed_mean, power.item())
    quasi_likelihood = residual / variance
    return torch.mean(quasi_likelihood)


class VAE(BaseMinifiedModeModuleClass):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.encode_covariates = encode_covariates
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.power = torch.nn.Parameter(torch.tensor(1.0))  # Initialize b as a learnable parameter

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_batch_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        n_input_decoder = n_latent + n_continuous_cov
        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )


    def _get_inference_input(self, tensors):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        x = tensors[REGISTRY_KEYS.X_KEY]
        input_dict = {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key]) if size_factor_key in tensors.keys() else None
        )
        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "size_factor": size_factor,
        }
        return input_dict

    def _compute_local_library_params(self, batch_index):
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(one_hot(batch_index, n_batch), self.library_log_means)
        local_library_log_vars = F.linear(one_hot(batch_index, n_batch), self.library_log_vars)
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded
        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}
        return outputs

    @auto_move_data
    def _cached_inference(self, qzm, qzv, observed_lib_size, n_samples=1):
        dist = Normal(qzm, qzv.sqrt())
        untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
        z = self.z_encoder.z_transformation(untran_z)
        library = torch.log(observed_lib_size)
        if n_samples > 1:
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )
        outputs = {"z": z, "qz_m": qzm, "qz_v": qzv, "ql": None, "library": library}
        return outputs

    @auto_move_data
    def generative(self, z, library, batch_index, cont_covs=None, cat_covs=None, size_factor=None, y=None, transform_batch=None):
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        if not self.use_size_factor_key:
            size_factor = library
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return {
            "power": self.power,
            "px_r": px_r,
            "px_scale": px_scale,
            "pl": pl,
            "pz": pz,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconstructed_mean = generative_outputs["px_r"]
        power = generative_outputs["power"]

        reconst_loss = quasi_likelihood_loss(reconstructed_mean, x, self.dispersion, power).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        loss = torch.mean(reconst_loss + weighted_kl_local)
        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local)

class CustomSCVI(SCVI):
    @property
    def _module_cls(self):
        return VAE
