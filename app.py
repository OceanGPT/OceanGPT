# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os
import random
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser


os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import gradio as gr

mllm = LLM(
    model="OceanGPT-V's path",
    max_model_len=4096,
    max_num_seqs=5,
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "fps": 1,
    }
)

llm = LLM(model="OceanGPT's path")

coder = LLM(model="OceanGPT-coder's path")

# Qwen2.5
def chat_qwen(questions: list[str],temperature: float, top_p: float, max_tokens: int):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outputs = llm.generate(questions, sampling_params)[0]
    generated_text = outputs.outputs[0].text
    return generated_text

def chat_qwen_coder(questions: list[str],temperature: float, top_p: float, max_tokens: int):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outputs = llm.generate(questions, sampling_params)[0]
    generated_text = outputs.outputs[0].text
    return generated_text
 
# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str):

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{question}<|im_end|>\n"
         "<|im_start|>assistant\n") for question in questions
    ]
    stop_token_ids = None
    return prompts, stop_token_ids


def get_multi_modal_input(img_questions,image_path):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
        # Input image and question
    image = Image.open(image_path).convert("RGB")

    return {
        "data": image,
        "questions": img_questions,
    }



def apply_image_repeat(image_repeat_prob, num_prompts, data,
                       prompts: list[str], modality):
    """Repeats images with provided probability of "image_repeat_prob". 
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        inputs.append({
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: cur_image
            }
        })

    return inputs


def chat_with_qwenvl(img_questions,image_path,temperature: float, top_p: float, max_tokens: int):
    modality = "image"
    mm_input = get_multi_modal_input(img_questions,image_path)
    data = mm_input["data"]
    questions = mm_input["questions"]

    prompts, stop_token_ids = run_qwen2_5_vl(questions, modality)
    prompts = [prompts[0]]

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)

    num_prompts = 1
    if num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {
                modality: data
            },
        }
    else:
        # Use the same image for all prompts
        inputs = [{
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: data
            },
        } for i in range(num_prompts)]


    outputs = mllm.generate(inputs, sampling_params=sampling_params)

    return outputs[0].outputs[0].text


def create_demo():
    with gr.Blocks() as demo:
        with gr.Tab("OceanGPT-V"):    
            with gr.Row():
                with gr.Column():
                    mllm_text = gr.Textbox(placeholder="Input text query", label="text input")
                    mllm_image = gr.Image(type="filepath", label="image input")
                    temperature = gr.Slider(minimum=0, maximum=2, label="temperature", step=0.1, value=1.2)
                    top_p = gr.Slider(minimum=0, maximum=1, label="top_p", step=0.01, value=0.95)
                    max_tokens = gr.Slider(minimum=1, maximum=1024, label="max_tokens", step=1, value=512)
                    clear_button = gr.ClearButton(components=[mllm_text, mllm_image],value="Clear")
                    run_botton = gr.Button("Run")
                with gr.Column():
                    response_res = gr.Textbox(label="OceanGPT-V's response")
            
            inputs = [mllm_text, mllm_image, temperature, top_p, max_tokens]
            outputs = [response_res]
            
            examples = [["Based on the observation of Figure 1 depicting a crucian carp from Loch Rannoch, how can the positioning and appearance of the scars described in the context be used to infer the size and shape of the hook used, and what does their placement suggest about the fish's behavior during its capture and escape?","case_0.png"],
                        ["Analyze the spatial distribution of the gradient vector field magnitudes illustrated in Fig. 6. How does the variability in color intensity across the figure inform the method of steepest descent, and what are the potential implications for understanding the underlying physical processes or geographic features represented?","case_23.png"],
                        ["Analyzing Figure 1, which depicts the New England and Corner Rise seamounts within the North Atlantic, could you elaborate on how the spatial distribution of these seamounts, as indicated by the yellow boxes, relates to the historical effects of bottom-trawling fisheries, and what implications this might have for future management strategies in these regions?","case_77.png"],
                        ["Considering the varying percentage coverage of Marine Protected Areas (MPAs) depicted in Figure 1, which includes both less protected and highly protected regions across different sectors, analyze why some areas like the Southern North Sea showcase significantly higher MPA coverage in comparison to the Northern North Sea. How does this distribution pattern reflect upon the consistency and potential efficacy of the MPAs in achieving marine conservation goals in UK waters?","case_94.png"]]
            
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                cache_examples=False,
                run_on_click=False
            )
            clear_button.add([response_res])
            run_botton.click(fn=chat_with_qwenvl,
                            inputs=inputs, outputs=outputs)
            
        with gr.Tab("OceanGPT"):
            with gr.Row():
                with gr.Column():
                    llm_text = gr.Textbox(placeholder="Input query", label="text input")
                    temperature = gr.Slider(minimum=0, maximum=2, label="temperature", step=0.1, value=1.2)
                    top_p = gr.Slider(minimum=0, maximum=1, label="top_p", step=0.01, value=0.95)
                    max_tokens = gr.Slider(minimum=1, maximum=1024, label="max_tokens", step=1, value=512)
                    clear_button = gr.ClearButton(components=[llm_text],value="Clear")
                    run_botton = gr.Button("Run")
                with gr.Column():
                    llm_response_res = gr.Textbox(label="OceanGPT's response")
            
            inputs = [llm_text, temperature, top_p, max_tokens]
            outputs = [llm_response_res]
            
            examples = [["Which fish species has been suggested as a potential ecological indicator due to its northward distribution with a 0.5oN change? (A) Tarphops oligolepis (B) Liachirus melanospilosa (C) Ostorhinchus fasciatus (D) Johnius taiwanensis"],
                        ["What impact does global warming have on marine life? (A) Promotes rapid growth (B) Changes migration patterns (C) Increases sea level (D) Reduces carbon dioxide concentration"],
                        ["What is the primary driver of thermohaline circulation? (A) Temperature and salinity (B) Ocean currents (C) Wind patterns	(D) Amount of sea ice"],
                        ["What is the effect of atmospheric forcing associated with warm inflow events?	(A) Enhanced absorption of sunlight (B) Increased ocean salinity (C) Weakening of coastal easterlies (D) Increased atmospheric pressure"]]
            
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                cache_examples=False,
                run_on_click=False
            )
            clear_button.add([response_res])
            run_botton.click(fn=chat_qwen,
                            inputs=inputs, outputs=outputs)
            
        with gr.Tab("OceanGPT-coder"):
            with gr.Row():
                with gr.Column():
                    llm_text = gr.Textbox(placeholder="Input query", label="text input")
                    temperature = gr.Slider(minimum=0, maximum=2, label="temperature", step=0.6, value=1.2)
                    top_p = gr.Slider(minimum=0, maximum=1, label="top_p", step=0.01, value=0.8)
                    max_tokens = gr.Slider(minimum=1, maximum=4096, label="max_tokens", step=512, value=2048)
                    clear_button = gr.ClearButton(components=[llm_text],value="Clear")
                    run_botton = gr.Button("Run")
                with gr.Column():
                    llm_response_res = gr.Textbox(label="OceanGPT-coder's response")
            
            inputs = [llm_text, temperature, top_p, max_tokens]
            outputs = [llm_response_res]
            
            examples = [["请为水下机器人生成MOOS代码，实现如下任务：按照顺序分别往以下几点 60,-40:60,-160:150,-160:180,-100:150,-40，速度为2m/s，任务执行两次，任务完成后返回原点。"],
                        ["请为水下机器人生成MOOS代码，实现如下任务：先前往3m的深度，然后按照顺序定高2m分别前往以下几点（10，20），（30，40），速度为2m/s，任务执行一次，完成后以定深1m返回原点，而后上浮。"],
                        ["请为水下机器人生成MOOS代码，实现如下任务：先回到（50,20）点，然后以（15,20）点为圆形，做半径为30的圆周运动，持续时间200s，速度4 m/s。"],
                        ["请为水下机器人生成MOOS代码，实现如下任务：先下潜10m深度，然后在3m深度，维持航向为180°，并持续3分钟，然后以1m/s的速度回到原点。"]]
            
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                cache_examples=False,
                run_on_click=False
            )
            clear_button.add([response_res])
            run_botton.click(fn=chat_qwen_coder,
                            inputs=inputs, outputs=outputs)
            
        with gr.Accordion("Disclaimer"):
            gr.Markdown("""
            This project is purely an academic exploration rather than a product(本项目仅为学术探索并非产品应用). Please be aware that due to the inherent limitations of large language models, there may be issues such as hallucinations.
            """)
    return demo

description = """"
# OceanGPT: The open source LLM and MLLM for ocean science.
**Note**: Due to network restrictions, it is recommended that the size of the uploaded image be less than **1M**.

Please refer to our [project](http://oceangpt.zjukg.cn/) for more details.
"""

with gr.Blocks(css="h1,p {text-align: center !important;}") as demo:
    gr.Markdown(description)
    create_demo()

demo.queue().launch(server_name="0.0.0.0",server_port=7860)