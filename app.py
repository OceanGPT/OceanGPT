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
import fitz
from docx import Document
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

def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""
    
# Qwen2.5
def chat_qwen(questions: list[str], llm_file, temperature: float, top_p: float, max_tokens: int):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    if llm_file and llm_file.lower().endswith((".pdf", ".docx")):
        text = extract_text_from_file(llm_file)
        questions = [text + '\n' + question for question in questions]
    outputs = llm.generate(questions, sampling_params)[0]
    generated_text = outputs.outputs[0].text
    return generated_text

def chat_qwen_coder(questions: list[str],temperature: float, top_p: float, max_tokens: int):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outputs = coder.generate(questions, sampling_params)[0]
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

def check_file_size(file):
    if file is None:
        return gr.update(visible=False)
    if isinstance(file, str):
        file_path = file
    elif isinstance(file, dict) and "name" in file:
        file_path = file.get("name")
    else:
        return gr.update(visible=False)
    size_in_bytes = os.path.getsize(file_path)
    if size_in_bytes > 1 * 1024 * 1024:
        return gr.update(value="⚠️ Uploaded file exceeds 1MB, please upload a smaller file.", visible=True)
    else:
        return gr.update(visible=False)

def create_demo():
    with gr.Blocks(css="""
        .textarea-auto-wrap textarea {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            resize: vertical !important;
        }
        .textarea-fixed-height textarea {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            resize: none !important;
            max-height: 600px !important;
        }
        .image-with-scroll {
            max-height: 400px !important;
            overflow: auto !important;
        }
        .image-with-scroll img {
            max-width: 100% !important;
            height: auto !important;
        }
    """) as demo:
        with gr.Tab("OceanGPT-o"):    
            with gr.Row():
                with gr.Column():
                    mllm_text = gr.TextArea(
                        placeholder="Input text query", 
                        label="text input", 
                        lines=3,
                        max_lines=15,
                        elem_classes=["textarea-auto-wrap"]
                    )
                    file_warning = gr.Markdown("", visible=False)
                    mllm_image = gr.Image(type="filepath", label="image input", elem_classes=["image-with-scroll"])
                    temperature = gr.Slider(minimum=0, maximum=2, label="temperature", step=0.1, value=0.6)
                    top_p = gr.Slider(minimum=0, maximum=1, label="top_p", step=0.01, value=1.0)
                    max_tokens = gr.Slider(minimum=1, maximum=4096, label="max_tokens", step=1, value=2048)
                    clear_button = gr.ClearButton(components=[mllm_text, mllm_image],value="Clear")
                    run_botton = gr.Button("Run")
                with gr.Column():
                    response_res = gr.TextArea(
                        label="OceanGPT-o's response", 
                        lines=8,
                        max_lines=20,
                        elem_classes=["textarea-fixed-height"]
                    )
            
            mllm_image.change(fn=check_file_size, inputs=mllm_image, outputs=[file_warning])
            
            inputs = [mllm_text, mllm_image, temperature, top_p, max_tokens]
            outputs = [response_res]
            
            examples = [
                    [
                        "作为海洋科学家，请分析所提供的声呐图像。描述在图像中检测到的物体，并尽可能详细地说明它们的位置。", "figs/case_0.png"
                    ],
                    [
                        "As a marine scientist, analyze the sonar images provided. Describe the objects detected in the images and specify their locations with as much detail as possible.", "figs/case_7.png"
                    ],
                    [
                        "请分析 CTD 投放站点（以空白方块和空心圆表示）的布局和分布，结合阿尔塔马哈河口海湾水流结构在横断面和沿河方向上的变化。这些站点的设置如何有助于获取关键数据，从而在河道不同区域上估算残余流和净输运？", "figs/case_1.png"
                    ],
                    [
                        "Analyze the arrangement and distribution of CTD casting stations (denoted by blank squares and open circles) in the context of the study's aim to understand cross- and along-channel variations in the current structures of the Altamaha River Sound. How do these station placements aid in capturing data essential for estimating the residual flow and net transport across different river sections?", "figs/case_1.png"
                    ]
                ]
            
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
                    llm_text = gr.TextArea(
                        placeholder="Input query", 
                        label="text input", 
                        lines=3,
                        max_lines=15,
                        elem_classes=["textarea-auto-wrap"]
                    )
                    file_warning = gr.Markdown("", visible=False)
                    llm_file = gr.File(label="Upload PDF / Word", file_types=[".pdf", ".docx"])
                    temperature = gr.Slider(minimum=0, maximum=2, label="temperature", step=0.1, value=0.6)
                    top_p = gr.Slider(minimum=0, maximum=1, label="top_p", step=0.01, value=1.0)
                    max_tokens = gr.Slider(minimum=1, maximum=4096, label="max_tokens", step=1, value=2048)
                    clear_button = gr.ClearButton(components=[llm_text, llm_file],value="Clear")
                    run_botton = gr.Button("Run")
                with gr.Column():
                    llm_response_res = gr.TextArea(
                        label="OceanGPT's response", 
                        lines=8,
                        max_lines=20,
                        elem_classes=["textarea-fixed-height"]
                    )
            
            llm_file.change(fn=check_file_size, inputs=llm_file, outputs=[file_warning])

            inputs = [llm_text, llm_file, temperature, top_p, max_tokens]
            outputs = [llm_response_res]
            
            examples = [
                    ["冷泉与热液喷口物种对比"],
                    ["如何区分自然形成的海底岩石与人造石质建筑遗迹？"],
                    ["Comparison between Cold Seep and Hydrothermal Vent Species"],
                    ["How to distinguish between naturally formed seafloor rocks and man-made stone architectural remains?"]
                ]
            
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                cache_examples=False,
                run_on_click=False
            )
            clear_button.add([llm_response_res])
            run_botton.click(fn=chat_qwen,
                            inputs=inputs, outputs=outputs)
            
        with gr.Tab("OceanGPT-coder"):
            with gr.Row():
                with gr.Column():
                    llm_text = gr.TextArea(
                        placeholder="Input query", 
                        label="text input", 
                        lines=3,
                        max_lines=15,
                        elem_classes=["textarea-auto-wrap"]
                    )
                    temperature = gr.Slider(minimum=0, maximum=2, label="temperature", step=0.1, value=0.6)
                    top_p = gr.Slider(minimum=0, maximum=1, label="top_p", step=0.01, value=1.0)
                    max_tokens = gr.Slider(minimum=1, maximum=4096, label="max_tokens", step=1, value=2048)

                    clear_button = gr.ClearButton(components=[llm_text],value="Clear")
                    run_botton = gr.Button("Run")
                with gr.Column():
                    llm_response_res = gr.TextArea(
                        label="OceanGPT-coder's response", 
                        lines=8,
                        max_lines=20,
                        elem_classes=["textarea-fixed-height"]
                    )
            
            inputs = [llm_text, temperature, top_p, max_tokens]
            outputs = [llm_response_res]
            
            examples = [
                    [
                        "请为水下机器人生成MOOS代码，执行声呐数据采集任务：按照顺序分别往以下几点 60,-40:60,-160:150,-160:180,-100:150,-40，速度为2m/s，任务执行两次，任务完成后返回原点。"
                    ],
                    [
                        "请为水下机器人生成MOOS代码，实现如下任务：先回到（0,0）点，然后以（0,0）点为圆形，做半径为50的圆周运动，持续时间120s。"
                    ]
                ]
            
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                cache_examples=False,
                run_on_click=False
            )
            clear_button.add([llm_response_res])
            run_botton.click(fn=chat_qwen_coder,
                            inputs=inputs, outputs=outputs)
            
        with gr.Accordion("Limitations"):
            gr.Markdown("""
            - The model may have hallucination issues.
            - Due to limited computational resources, OceanGPT-o currently only supports natural language generation for certain types of sonar images and ocean science images. OceanGPT-coder currently only supports MOOS code generation.
            - We did not optimize the identity and the model may generate identity information similar to that of Qwen/MiniCPM/LLaMA/GPT series models.
            - The model's output is influenced by prompt tokens, which may result in inconsistent results across multiple attempts.
            """)
    return demo

description = """
# Ocean Foundation Models
Upload documents (Word, PDF, or images) to help OceanGPT provide more accurate answers.

Please refer to our [project](http://www.oceangpt.blue/) for more details.
"""

with gr.Blocks(css="""
    h1,p {text-align: center !important;}
    .textarea-auto-wrap textarea {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }
    .textarea-fixed-height textarea {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        resize: none !important;
        max-height: 600px !important;
    }
    .image-with-scroll {
        max-height: 400px !important;
        overflow: auto !important;
    }
    .image-with-scroll img {
        max-width: 100% !important;
        height: auto !important;
    }
""") as demo:
    gr.Markdown(description)
    create_demo()

demo.queue().launch(server_name="0.0.0.0",server_port=7860)
