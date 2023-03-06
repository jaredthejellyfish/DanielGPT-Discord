import discord
from ..JellyHostML import JellyHostML

jhml = JellyHostML("1")


def make_embed(prompt, width, height, inference_steps, guideance_scale, negative_prompt, seed):
    _, inference_steps, guideance_scale, _, height, width, seed = jhml._make_input_safe_generate(
        prompt, inference_steps, guideance_scale, negative_prompt, height, width, seed)

    embed = discord.Embed(title="DanielGPT - StableDiffusionAPI",
                          description=f"{prompt}")
    embed.add_field(name="Width", value=width, inline=True)
    embed.add_field(name="Height", value=height, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="Inference Steps", value=inference_steps, inline=True)
    embed.add_field(name="Guideance Scale", value=guideance_scale, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)
    embed.add_field(name="Negative Prompt", value=negative_prompt, inline=True)
    embed.add_field(name="Seed", value=seed, inline=False)

    return embed


def make_embed_im2(prompt, strength, num_inference_steps, guidance_scale, negative_prompt):
    _, strength, num_inference_steps, guidance_scale, _ = jhml._make_input_safe_img2img(
        prompt, strength, num_inference_steps, guidance_scale, negative_prompt)

    embed = discord.Embed(title="DanielGPT - Img2ImgAPI",
                          description=f"{prompt}")
    embed.add_field(name="Strength", value=strength, inline=True)
    embed.add_field(name="Inference Steps",
                    value=num_inference_steps, inline=True)
    embed.add_field(name="Guideance Scale", value=guidance_scale, inline=True)
    embed.add_field(name="Negative Prompt", value=negative_prompt, inline=True)

    return embed
