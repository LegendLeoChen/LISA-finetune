from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def iou_loss(inputs, targets, num_masks, eps=1e-6):
    """
    IoU (Jaccard) Loss for binary mask segmentation.
    新增IoU损失
    Args:
        inputs: logits tensor, shape [B, H, W] or [B, 1, H, W]
        targets: ground truth mask, same shape as inputs
        num_masks: normalization factor, usually number of masks in batch
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)

    intersection = (inputs * targets).sum(-1)
    union = inputs.sum(-1) + targets.sum(-1) - intersection

    iou = (intersection + eps) / (union + eps)
    loss = 1 - iou
    return loss.sum() / (num_masks + 1e-8)


def boundary_loss(inputs, targets, num_masks, eps=1e-6):
    """
    Boundary loss using Sobel operator to extract edges. 
    边界损失，使用卷积算子求边缘，计算两个mask边缘L1损失。
    Args:
        inputs: logits tensor, shape [B, H, W] or [B, 1, H, W]
        targets: ground truth mask, same shape as inputs
        num_masks: normalization factor, usually number of masks in batch
    """
    if inputs.shape[0] == 0:
        return torch.tensor(0.0, device=inputs.device, requires_grad=True)
    inputs = inputs.sigmoid()

    # Sobel filter for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=inputs.dtype, device=inputs.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1], [0,0,0], [1,2,1]], dtype=inputs.dtype, device=inputs.device).view(1,1,3,3)

    # ➔ groups=inputs.shape[1] 逐通道卷积
    edge_pred_x = F.conv2d(inputs, sobel_x.expand(inputs.shape[0], 1, 3, 3), padding=1, groups=inputs.shape[0])
    edge_pred_y = F.conv2d(inputs, sobel_y.expand(inputs.shape[0], 1, 3, 3), padding=1, groups=inputs.shape[0])
    edge_pred = torch.sqrt(edge_pred_x**2 + edge_pred_y**2 + eps)

    targets = targets.float()
    edge_gt_x = F.conv2d(targets, sobel_x.expand(targets.shape[0], 1, 3, 3), padding=1, groups=targets.shape[0])
    edge_gt_y = F.conv2d(targets, sobel_y.expand(targets.shape[0], 1, 3, 3), padding=1, groups=targets.shape[0])
    edge_gt = torch.sqrt(edge_gt_x**2 + edge_gt_y**2 + eps)

    loss = F.l1_loss(edge_pred, edge_gt, reduction="none")
    loss = loss.flatten(1).mean(1)       # → shape [1, 3]
    loss = loss.mean()                   # → scalar
    return loss

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class MultiLayerTextEncoder(nn.Module):
    def __init__(self, selected_layers, hidden_size, out_dim, dropout=0.1):
        """
        selected_layers: List[int], 1-based layer numbers, e.g., [29, 31, 33]
        hidden_size: input hidden dimension (from LLaMA)
        out_dim: desired output dim
        """
        super().__init__()
        self.selected_layers = selected_layers
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                # nn.GELU(),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, out_dim),
            )
            for _ in selected_layers
        ])
        
        self.layer_weights = nn.Parameter(torch.ones(len(selected_layers))) # 可学习权重
        # 初始化所有参数
        self._init_weights()

    def _init_weights(self):
        for fc_block in self.fcs:
            for layer in fc_block:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        # 初始化层融合权重为均匀分布（避免训练初期偏置某一层）
        nn.init.constant_(self.layer_weights, 1.0 / len(self.selected_layers))
        
    def forward(self, all_hidden_states):
        """
        all_hidden_states: List of [B, T, D], length = num_total_layers (e.g., 33)
        Returns:
            fused: [B, T, out_dim]
        """
        outputs = []
        for i, layer_id in enumerate(self.selected_layers):
            # Convert from 1-based to 0-based (layer 1 = last layer)
            x = all_hidden_states[-layer_id]  # [B, T, D]
            outputs.append(self.fcs[i](x))      # [B, T, out_dim]

        # average fusion
        stacked = torch.stack(outputs, dim=0)  # [L, B, T, D]
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        fused = (weights * stacked).sum(dim=0)          # 加权求和
        return fused

class ImageFeatureProjector(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048, output_channels=256, seq_h=16, seq_w=16, out_h=64, out_w=64):
        super().__init__()

        self.seq_h = seq_h
        self.seq_w = seq_w
        self.out_h = out_h
        self.out_w = out_w
        self.output_channels = output_channels

        # 先用 MLP 把每个 token 的特征变成 output_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_channels)
        )

        # 用卷积上采样：16x16 -> 64x64
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=4)  # 16x16 -> 64x64
        )

    def forward(self, x):
        # x: [n, 256, 4096]
        n, seq, dim = x.shape
        # assert seq == self.seq_h * self.seq_w, f"输入序列长度应为 {self.seq_h*self.seq_w}, 当前是 {seq}"

        x = self.mlp(x)          # -> [n, 256, output_channels]
        x = x.mean(0, keepdim=True)  # -> [1, 256, output_channels]
        x = x.transpose(1, 2)    # -> [1, output_channels, 256]
        x = x.reshape(1, self.output_channels, self.seq_h, self.seq_w)  # -> [1, C, 16, 16]
        x = self.upsample(x)     # -> [1, C, 64, 64]

        return x
    
class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        
        # 原版
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        
        # 多隐藏特征融合
        self.text_hidden_fcs1 = MultiLayerTextEncoder(
            selected_layers=[15, 10, 5, 3, 1],      # 最后第n层
            hidden_size=in_dim,               # 视具体模型而定（LLaMA-7B 是 4096）
            out_dim=out_dim,                    # 你需要的输出维度
            dropout=0.1
        )
        
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        
        self.projector = ImageFeatureProjector()


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.iou_loss_weight = kwargs.pop("iou_loss_weight", None)
        self.boundary_loss_weight = kwargs.pop("boundary_loss_weight", None)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        feedback: bool = True,
        enhance_mlp: bool = True,
        vision_mix: bool = True,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            # output_hidden_states 是 list of tuple (每个batch一个hidden_states tuple)
            output_hidden_states_batches = []

            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i, image_features = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states_batches.append(output_i.hidden_states)  # tuple of layers
                if vision_mix:
                    image_embeddings[i] = 0.3 * self.model.projector(image_features) + 0.7 * image_embeddings[i]
                torch.cuda.empty_cache()

            # 现在 output_hidden_states_batches 是一个 list，里面每个元素是一个 hidden_states 的 tuple（长度为 num_layers）

            # 将每层的结果拼接：output_hidden_states[layer] 结构对齐训练模式
            num_layers = len(output_hidden_states_batches[0])  # 通常是13或25层
            output_hidden_states = []

            for layer_idx in range(num_layers):
                # 取出所有 batch 在该层的输出，并拼接
                layer_outputs = [batch_hidden[layer_idx] for batch_hidden in output_hidden_states_batches]
                merged_layer = torch.cat(layer_outputs, dim=0)  # 沿 batch 维拼接
                output_hidden_states.append(merged_layer)

            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output, image_features = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
            if vision_mix:
                for i in range(len(offset) - 1):
                    start, end = offset[i], offset[i + 1]
                    chunk = image_features[start:end]  # [n_i, 256, 4096]
                    image_embeddings[i] = 0.3 * self.model.projector(chunk) + 0.7 * image_embeddings[i]      # -> [1, C, H, W]

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        
        if not enhance_mlp:
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))     # 原版
        else:
            hidden_states.append(self.model.text_hidden_fcs1(output_hidden_states))       # 多隐藏特征融合

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks, low_masks = [], []
        for i in range(len(pred_embeddings)):
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            low_masks.append(low_res_masks[:, 0])
            pred_masks.append(pred_mask[:, 0])
        
        if feedback:
            pred_masks = []
            for i in range(len(pred_embeddings)):
                sparse_embeddings2, dense_embeddings2 = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=low_masks[i].to(pred_embeddings[i].dtype).unsqueeze(1),
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings2 = sparse_embeddings2.to(pred_embeddings[i].dtype)
                low_res_masks2, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings2,
                    dense_prompt_embeddings=dense_embeddings2,
                    multimask_output=multimask_output,
                )
                pred_mask2 = self.model.visual_model.postprocess_masks(
                    low_res_masks2,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask2[:, 0])
        
        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        mask_iou_loss = 0
        mask_boundary_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_iou_loss += (
                iou_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_boundary_loss += (
                boundary_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_iou_loss = self.iou_loss_weight * mask_iou_loss / (num_masks + 1e-8)
        mask_boundary_loss = self.boundary_loss_weight * mask_boundary_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss + mask_iou_loss + mask_boundary_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_iou_loss": mask_iou_loss,
            "mask_boundary_loss": mask_boundary_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):  
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
