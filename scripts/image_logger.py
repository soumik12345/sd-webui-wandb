import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

import wandb


class ImageLogger(scripts.Script):
    def title(self):
        return "Log to Weights & Biases"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        gr.Markdown("## Logging to Weights & Biases!")
        return super().ui(is_img2img)

    def run(self, p):
        wandb.init(
            project=opts.wandb_project,
            entity=opts.wandb_entity,
            job_type=self.job_type,
        )

        proc = process_images(p)

        config = wandb.config
        config.prompt = proc.prompt
        config.negative_prompt = proc.negative_prompt
        config.all_prompts = proc.all_prompts
        config.all_negative_prompts = proc.all_negative_prompts
        config.seed = proc.seed
        config.subseed = proc.subseed
        config.all_seeds = proc.all_seeds
        config.all_subseeds = proc.all_subseeds
        config.subseed_strength = proc.subseed_strength
        config.info = proc.info
        config.comments = proc.comments
        config.width = proc.width
        config.height = proc.height
        config.sampler_name = proc.sampler_name
        config.cfg_scale = proc.cfg_scale
        config.image_cfg_scale = proc.image_cfg_scale
        config.steps = proc.steps
        config.batch_size = proc.batch_size
        config.restore_faces = proc.restore_faces
        config.seed_resize_from_w = proc.seed_resize_from_w
        config.seed_resize_from_h = proc.seed_resize_from_h
        config.denoising_strength = proc.denoising_strength
        config.extra_generation_params = proc.extra_generation_params
        config.index_of_first_image = proc.index_of_first_image
        config.styles = proc.styles
        config.job_timestamp = proc.job_timestamp
        config.clip_skip = proc.clip_skip
        config.eta = proc.eta
        config.ddim_discretize = proc.ddim_discretize
        config.s_churn = proc.s_churn
        config.s_tmin = proc.s_tmin
        config.s_tmax = proc.s_tmax
        config.s_noise = proc.s_noise
        config.s_min_uncond = proc.s_min_uncond
        config.sampler_noise_scheduler_override = proc.sampler_noise_scheduler_override
        config.is_using_inpainting_conditioning = proc.is_using_inpainting_conditioning
        config.token_merging_ratio = proc.token_merging_ratio
        config.token_merging_ratio_hr = proc.token_merging_ratio_hr
        config.infotexts = proc.infotexts

        wandb_table = wandb.Table(
            columns=[
                "Prompt",
                "Negative-Prompt",
                "Generated-Image",
                "Image-Size",
                "Sampler-Name",
                "Seed",
            ]
        )

        for image in proc.images:
            wandb_image = wandb.Image(image)
            wandb_table.add_data(
                config.prompt,
                config.negative_prompt,
                wandb_image,
                {"Height": config.height, "Width": config.width},
                config.sampler_name,
                config.seed,
            )
            wandb.log({"Generated-Images": wandb_image})
        wandb.finish()

        return proc
