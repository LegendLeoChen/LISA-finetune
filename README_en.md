# 🚀 Overview
[中文](README.md)

Based on [LISA](https://github.com/dvlab-research/LISA), this project implements fine-tuning, prediction, and evaluation for segmentation tasks using the dataset [LabPicsV1](https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1). Configuration methods are provided below, and you can also refer to the [old documentation](README_origin.md).

![1](./vis_output/my/tower_masked_img_0.jpg)

![2](./vis_output/my/1.png)

# 🚀 Configuration

Install libraries (you can start with a virtual environment). Most library versions need to be correct; otherwise, errors are likely, especially for `transformers`, `gradio`, `numpy`, etc. If other libraries fail, try installing newer versions.

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

# 🚀 Download Weights

Download the weights. The following are relatively new and you can choose either 7B or 13B (this article uses 7B because the 4090D memory is just not enough to run bf16 13B), and place them in a custom location. LISA should directly include LLaVA and SAM, which means the entirety of the network structure. If you only need prediction and evaluation, you only need to download LISA and the LLaVA visual backbone (theoretically, the second one doesn't need to be downloaded, but the code seems to read the weights separately); if you want to fine-tune, you will need LISA (or LLaVA), LLaVA visual backbone, and SAM-VIT-H.

- LISA: [Hugging Face](https://huggingface.co/xinlai/LISA-7B-v1)
- LLaVA Visual Backbone: [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14)
- LLaVA: [Hugging Face](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1)
- SAM-VIT-H: [Direct Link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

After downloading, open the `config.json` file in the LISA weights folder, and change the path of `vision_tower` to the path of the visual backbone folder.

# 🚀 Project Structure

The following are the parts that have been added or modified:
- `myutils`:
    - `dataset.py`: The dataset class for LabPicsV1;
    - `metric.py`: Evaluation metrics, including IoU, GIoU, CIoU, Dice, PA, Boundary F1;
- `merge_lora_weights_and_save_hf_model.py`: Convert the fine-tuned weights into full LISA weights;
- `test.py`: Evaluate the model on the LabPicsV1 dataset using metrics from `metric.py`;
- `train.py`: Train LISA using the LabPicsV1 dataset.

# 🚀 Usage

- Chat and Prediction: Run `chat.py` or `app.py`
- Evaluation: Run `test.py`
- Fine-tuning:
    1. Run `train.py` to fine-tune;
    2. Run `cd ./runs/lisa/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin` to get the full LISA weights;
    3. Run `merge_lora_weights_and_save_hf_model.py` to merge LoRA weights and obtain weights in Hugging Face format;