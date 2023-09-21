import gradio as gr
import os

from modules import shared
from modules import script_callbacks


def wandb_settings():
    section = ("wandb", "Weights & Biases")
    shared.opts.add_option(
        "wandb_api_key",
        shared.OptionInfo(
            default="",
            label="Weights & Biases API Key",
            component=gr.Textbox,
            component_args={"type": "password", "interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "wandb_project",
        shared.OptionInfo(
            default="",
            label="Weights & Biases Project",
            component=gr.Textbox,
            component_args={"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "wandb_entity",
        shared.OptionInfo(
            default="",
            label="Weights & Biases Entity",
            component=gr.Textbox,
            component_args={"interactive": True},
            section=section,
        ),
    )


script_callbacks.on_ui_settings(wandb_settings)
