import modules.scripts as scripts
import gradio as gr
import os

from modules.processing import process_images, Processed
from modules.processing import Processed, StableDiffusionProcessing
from modules.shared import opts, cmd_opts, state, log

import wandb


def login_to_wandb():
    try:
        api_key = str(opts.wandb_api_key)
        if api_key != "":
            os.environ["WANDB_API_KEY"] = api_key
            log.info("Successfully set WandB API Key")
        else:
            log.error("Unable to log in to WandB")
    except AttributeError:
        log.error("Unable to log in to WandB")


class ImageLogger(scripts.Script):
    def title(self):
        return "Log to Weights & Biases"

    def show(self, is_img2img):
        self.job_type = "img2img" if is_img2img else "txt2img"
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Weights & Biases", open=False):
            wandb_workspace_url = (
                f"https://wandb.ai/{opts.wandb_entity}/{opts.wandb_project}"
            )
            gr.Markdown(
                f"Logging to project [**{opts.wandb_project}/{opts.wandb_entity}**]({wandb_workspace_url})"
            )
        return super().ui(is_img2img)
    
    def _get_wandb_table_columns(self):
        if self.job_type == "txt2img":
            return [
                    "Prompt",
                    "Negative-Prompt",
                    "Generated-Image",
                    "Image-Size",
                    "Sampler-Name",
                    "Seed",
                ]
        else:
            return []
    
    def _set_config(self, p: StableDiffusionProcessing, processed: Processed):
        config = wandb.config
        config.prompt = processed.prompt
        config.negative_prompt = processed.negative_prompt
        config.all_prompts = processed.all_prompts
        config.all_negative_prompts = processed.all_negative_prompts
        config.seed = processed.seed
        config.subseed = processed.subseed
        config.all_seeds = processed.all_seeds
        config.all_subseeds = processed.all_subseeds
        config.subseed_strength = processed.subseed_strength
        config.info = processed.info
        config.comments = processed.comments
        config.width = processed.width
        config.height = processed.height
        config.sampler_name = processed.sampler_name
        config.cfg_scale = processed.cfg_scale
        config.image_cfg_scale = processed.image_cfg_scale
        config.steps = processed.steps
        config.batch_size = processed.batch_size
        config.restore_faces = processed.restore_faces
        config.seed_resize_from_w = processed.seed_resize_from_w
        config.seed_resize_from_h = processed.seed_resize_from_h
        config.denoising_strength = processed.denoising_strength
        config.extra_generation_params = processed.extra_generation_params
        config.index_of_first_image = processed.index_of_first_image
        config.styles = processed.styles
        config.job_timestamp = processed.job_timestamp
        config.clip_skip = processed.clip_skip
        config.eta = processed.eta
        config.ddim_discretize = processed.ddim_discretize
        config.s_churn = processed.s_churn
        config.s_tmin = processed.s_tmin
        config.s_tmax = processed.s_tmax
        config.s_noise = processed.s_noise
        config.s_min_uncond = processed.s_min_uncond
        config.sampler_noise_scheduler_override = (
            processed.sampler_noise_scheduler_override
        )
        config.is_using_inpainting_conditioning = (
            processed.is_using_inpainting_conditioning
        )
        config.token_merging_ratio = processed.token_merging_ratio
        config.token_merging_ratio_hr = processed.token_merging_ratio_hr
        config.infotexts = processed.infotexts
        
        return config

    def postprocess(self, p: StableDiffusionProcessing, processed: Processed, *args):
        login_to_wandb()
        with wandb.init(
            project=opts.wandb_project, entity=opts.wandb_entity, job_type=self.job_type
        ):
            config = self._set_config(p, processed)
            wandb_table = wandb.Table(columns=self._get_wandb_table_columns())

            for image in processed.images:
                wandb_image = wandb.Image(image)
                row = []
                if self.job_type == "txt2img":
                    row += [
                        config.prompt,
                        config.negative_prompt,
                        wandb_image,
                        {"Height": config.height, "Width": config.width},
                        config.sampler_name,
                        config.seed,
                    ]
                wandb_table.add_data(*row)
                wandb.log({"Generated-Images": wandb_image})
        
            if self.job_type == "txt2img":
                wandb.log({"Text-to-Image": wandb_table})

        return processed
