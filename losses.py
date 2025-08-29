# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self._printed_feature_shapes = True

    def forward(self, outputs):
        image_embed = outputs["image_embed"]
        text_embed = outputs["text_embed"]
        logit_scale = outputs["logit_scale"]
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = utils.all_gather_batch(
            [image_embed, text_embed]
        )

        if self._printed_feature_shapes:
            print(f"[DEBUG] Multi-node gathered feature shapes:")
            print(f"  image_embed_all: {image_embed_all.shape}")
            print(f"  text_embed_all: {text_embed_all.shape}")
            self._printed_feature_shapes = False

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        ) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {"loss": loss, "clip_loss": loss, "clip_acc": acc}


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None
        self._printed_feature_shapes = True

    def forward(self, outputs):
        q_a = outputs["aug1_embed"]
        q_b = outputs["aug2_embed"]

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if self._printed_feature_shapes:
            print(f"[DEBUG] Multi-node gathered feature shapes:")
            print(f"  k_a: {k_a.shape}")
            print(f"  k_b: {k_b.shape}")
            self._printed_feature_shapes = False

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {"loss": loss, "ssl_loss": loss, "ssl_acc": acc}


class SLIPLoss(nn.Module):
    def __init__(self, ssl_loss, ssl_scale):
        super().__init__()
        self.clip_loss = CLIPLoss()
        self.ssl_loss = ssl_loss
        self.ssl_scale = ssl_scale

    def forward(self, outputs):
        clip_loss_dict = self.clip_loss(outputs)
        clip_loss = clip_loss_dict["clip_loss"]
        clip_acc = clip_loss_dict["clip_acc"]

        ssl_loss_dict = self.ssl_loss(outputs)
        ssl_loss = ssl_loss_dict["ssl_loss"]
        ssl_acc = ssl_loss_dict["ssl_acc"]

        return {
            "loss": clip_loss + self.ssl_scale * ssl_loss,
            "clip_loss": clip_loss,
            "clip_acc": clip_acc,
            "ssl_loss": ssl_loss,
            "ssl_acc": ssl_acc,
        }


class TeMoLoss(nn.Module):

    def __init__(
        self,
        tau_min=0.01,
        tau_alpha=0.04,
        total_steps=1,
    ):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

        # TeMo params
        self.tau_min = tau_min
        self.tau_alpha = tau_alpha
        self.total_steps = total_steps

        # For debugging
        self._printed_feature_shapes = True

        print(f"Total number of steps: {self.total_steps}")
        print(f"Tau min: {self.tau_min}")
        print(f"Tau alpha: {self.tau_alpha}")

    def _sim_to_temperature(self, sim):
        return self.tau_min + self.tau_alpha * torch.sqrt((sim + 1.0) / 2.0)

    def _compute_alpha_and_beta(self, current_step):
        normalized_current_step = current_step / self.total_steps
        alpha = (normalized_current_step - 1.0) ** 2
        beta = normalized_current_step**2
        return alpha, beta

    def forward(
        self,
        image_features,
        text_features,
        image_aug_features,
        text_aug_features,
        logit_scale,
        current_step,
    ):

        if self.world_size > 1:
            all_image_features, all_text_features = utils.all_gather_batch(
                [image_features, text_features]
            )

            all_image_aug_features, all_text_aug_features = utils.all_gather_batch(
                [image_aug_features, text_aug_features]
            )

            if self._printed_feature_shapes:
                print(f"[DEBUG] Multi-node gathered feature shapes:")
                print(f"  all_image_features: {all_image_features.shape}")
                print(f"  all_text_features: {all_text_features.shape}")
                print(f"  all_image_aug_features: {all_image_aug_features.shape}")
                print(f"  all_text_aug_features: {all_text_aug_features.shape}")
                print(f"  world_size: {self.world_size}")
                self._printed_feature_shapes = False
        else:
            all_image_features = image_features
            all_text_features = text_features
            all_image_aug_features = image_aug_features
            all_text_aug_features = text_aug_features

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        device = image_features.device

        i2t_sim = all_image_features @ all_text_features.T
        t2i_sim = i2t_sim.T

        # ============== Compute normal CLIP loss ==================
        info_nce_loss = (
            F.cross_entropy(i2t_sim / logit_scale, self.labels)
            + F.cross_entropy(t2i_sim / logit_scale, self.labels)
        ) / 2
        # ============== Compute normal CLIP loss ==================

        # ============== Compute modulated CLIP loss ==================
        i2t_temp = self._sim_to_temperature(i2t_sim)
        t2i_temp = self._sim_to_temperature(t2i_sim)

        m_info_nce_loss = (
            F.cross_entropy(i2t_sim / i2t_temp, self.labels)
            + F.cross_entropy(t2i_sim / t2i_temp, self.labels)
        ) / 2
        # ============== Compute modulated CLIP loss ==================

        # ============== Compute modulated I2I loss ==================
        i2i_sim = all_image_features @ all_image_aug_features.T
        i2i_temp = self._sim_to_temperature(i2i_sim)

        m_i2i_loss = F.cross_entropy(i2i_sim / i2i_temp, self.labels)
        # ============== Compute modulated I2I loss ==================

        # ============== Compute modulated T2T loss ==================
        t2t_sim = all_text_features @ all_text_aug_features.T
        t2t_temp = self._sim_to_temperature(t2t_sim)

        m_t2t_loss = F.cross_entropy(t2t_sim / t2t_temp, self.labels)
        # ============== Compute modulated T2T loss ==================

        alpha, beta = self._compute_alpha_and_beta(current_step)

        total_loss = alpha * info_nce_loss + beta * (
            m_info_nce_loss + m_i2i_loss + m_t2t_loss
        )

        other_metrics = {
            "info_nce_loss": info_nce_loss,
            "m_info_nce_loss": m_info_nce_loss,
            "m_i2i_loss": m_i2i_loss,
            "m_t2t_loss": m_t2t_loss,
            "alpha": torch.tensor(alpha),
            "beta": torch.tensor(beta),
            **get_temperature_statistics(i2t_temp, "i2t/"),
            **get_temperature_statistics(t2i_temp, "t2i/"),
            **get_temperature_statistics(i2i_temp, "i2i/"),
            **get_temperature_statistics(t2t_temp, "t2t/"),
        }

        return {"loss": total_loss}, other_metrics


def get_temperature_statistics(per_sample_temperature, prefix="train/"):
    min_per_sample_temperature = per_sample_temperature.min()
    max_per_sample_temperature = per_sample_temperature.max()
    avg_per_sample_temperature = per_sample_temperature.mean()
    median_per_sample_temperature = per_sample_temperature.median()
    quantile_0_5_per_sample_temperature = per_sample_temperature.float().quantile(0.5)

    temp_of_positives = per_sample_temperature.diag()
    positive_samples_min_temperature = temp_of_positives.min()
    positive_samples_max_temperature = temp_of_positives.max()
    positive_samples_avg_temperature = temp_of_positives.mean()
    positive_samples_median_temperature = temp_of_positives.median()
    positive_samples_quantile_0_5_temperature = temp_of_positives.float().quantile(0.5)

    temp_of_negatives = per_sample_temperature[
        ~torch.eye(per_sample_temperature.size(0), dtype=bool)
    ]
    negative_samples_min_temperature = temp_of_negatives.min()
    negative_samples_max_temperature = temp_of_negatives.max()
    negative_samples_avg_temperature = temp_of_negatives.mean()
    negative_samples_median_temperature = temp_of_negatives.median()
    negative_samples_quantile_0_5_temperature = temp_of_negatives.float().quantile(0.5)

    return {
        f"{prefix}min_per_sample_temperature": min_per_sample_temperature,
        f"{prefix}max_per_sample_temperature": max_per_sample_temperature,
        f"{prefix}avg_per_sample_temperature": avg_per_sample_temperature,
        f"{prefix}median_per_sample_temperature": median_per_sample_temperature,
        f"{prefix}quantile_0.5_per_sample_temperature": quantile_0_5_per_sample_temperature,
        f"{prefix}positive_samples_min_temperature": positive_samples_min_temperature,
        f"{prefix}positive_samples_max_temperature": positive_samples_max_temperature,
        f"{prefix}positive_samples_avg_temperature": positive_samples_avg_temperature,
        f"{prefix}positive_samples_median_temperature": positive_samples_median_temperature,
        f"{prefix}positive_samples_quantile_0.5_temperature": positive_samples_quantile_0_5_temperature,
        f"{prefix}negative_samples_min_temperature": negative_samples_min_temperature,
        f"{prefix}negative_samples_max_temperature": negative_samples_max_temperature,
        f"{prefix}negative_samples_avg_temperature": negative_samples_avg_temperature,
        f"{prefix}negative_samples_median_temperature": negative_samples_median_temperature,
        f"{prefix}negative_samples_quantile_0.5_temperature": negative_samples_quantile_0_5_temperature,
    }
