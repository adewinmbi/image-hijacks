from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Type, Union
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, Int64, Bool
from torch import Tensor
from torch.nn import CrossEntropyLoss
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, ProcessorMixin
from tqdm import tqdm

from image_hijacks.models import AbstractLensModel
from image_hijacks.utils import PROJECT_ROOT, detach_numpy, load_model_with_cache, get_full_attention_mask

CACHE_DIR = Path('/home/adewinmb/orcd/scratch')

class Qwen3Lens(AbstractLensModel):
    """Class that adds additional functionality for experimenting with Qwen3-VL"""
    
    def __init__(
        self,
        model: Qwen3VLForConditionalGeneration,
        processor: ProcessorMixin,
        model_dtype: torch.dtype,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_dtype = model_dtype
        self._last_image_grid_thw = None  # Store grid info for Qwen3-VL's dynamic resolution
        
    def input_image_dims(self) -> Tuple[int, int]:
        """Returns (h, w) of input image
        Qwen3-VL uses dynamic resolution, but we'll return a reasonable default
        """
        # Qwen3-VL uses flexible image sizes, default to 448x448
        return (448, 448)
    
    def preprocess_image(
        self, img: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[Float[Tensor, "b c h w"], Optional[Bool[Tensor, "b img_seq_len"]]]:
        """Converts PIL image to unstandardized tensor with pixel values in [0, 1]"""
        if isinstance(img, Image.Image):
            img = [img]
        
        # Qwen3-VL's processor handles image preprocessing
        # We'll process them without normalization first
        print("Image: ", img)
        processed = self.processor(
            text="",
            images=img,
            return_tensors="pt",
        )
        
        pixel_values = processed["pixel_values"].to(self.model_dtype).to(self.device)
        print("Pixel values: ", pixel_values.shape)
        
        # Store image_grid_thw for later use in forward passes
        # This encodes temporal, height, width info for dynamic resolution
        if "image_grid_thw" in processed:
            self._last_image_grid_thw = processed["image_grid_thw"].to(self.device)
        
        # For attention mask, we'll return None as Qwen3 handles this internally
        return pixel_values, None
    
    def normalize_image(
        self, pixel_values: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        """Normalise batch of images
        Qwen3-VL processor already handles normalization, so this is a pass-through
        """
        # The processor handles normalization internally
        return pixel_values
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        mode: Literal["encoder", "decoder", "no_special_tokens"],
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        randomly_sample_system_prompt: bool = False,
    ) -> Tuple[Int64[Tensor, "b max_seq_len"], Bool[Tensor, "b max_seq_len"]]:
        """Given text or a list of text, returns batched tokenised text along with a padding mask"""
        if isinstance(text, str):
            text = [text]
        
        # Qwen3 is a decoder-only model
        if mode == "encoder":
            # For encoder mode, we tokenize with image placeholder
            tokenized = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
        elif mode == "decoder":
            # For decoder mode, tokenize without special tokens at start
            tokenized = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            # Add BOS token
            b, _ = tokenized["input_ids"].shape
            device = tokenized["input_ids"].device
            bos_id = self.processor.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = self.processor.tokenizer.eos_token_id  # Fallback
            tokenized["input_ids"] = torch.cat(
                [torch.full((b, 1), bos_id, device=device), tokenized["input_ids"]], dim=1
            )
            tokenized["attention_mask"] = torch.cat(
                [torch.full((b, 1), 1, device=device, dtype=torch.bool), tokenized["attention_mask"]],
                dim=1,
            )
        elif mode == "no_special_tokens":
            tokenized = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        return input_ids, attention_mask
    
    def to_string(
        self, tokens: Int64[Tensor, "b seq_len"], skip_special_tokens=True
    ) -> List[str]:
        """Given a batch of sequences of tokens, detokenise each sequence."""
        np_tokens = detach_numpy(tokens)
        return self.processor.tokenizer.batch_decode(
            sequences=np_tokens,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    
    def get_image_embeddings(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Optional[Float[Tensor, "b tok_seq_len h_lm"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Float[Tensor, "b img_seq_len h_lm"]:
        """Given a batch of unnormalised input images, return image embeddings."""
        # Qwen3-VL's vision model processes images
        # This is a simplified version - the actual implementation is more complex
        # due to Qwen3's dynamic resolution handling
        raise NotImplementedError(
            "Image embeddings extraction not yet fully implemented for Qwen3-VL. "
            "Use get_embeddings_from_image_and_tokens or generate_end_to_end instead."
        )
    
    def get_token_embeddings(
        self, tokens: Int64[Tensor, "b max_seq_len"]
    ) -> Float[Tensor, "b max_seq_len h_lm"]:
        """Given a batch of padded tokens, returns language model embeddings."""
        return self.model.model.embed_tokens(tokens).to(self.model_dtype)
    
    def get_embeddings_from_image_and_tokens(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b seq_len h_lm"], Int64[Tensor, "b seq_len"]]:
        """Given pixel values and input tokens, returns input embeddings and attention mask."""
        # For Qwen3-VL, the model handles the fusion of image and text internally
        # This is a complex operation that involves the vision transformer and cross-attention
        raise NotImplementedError(
            "Embedding extraction not yet fully implemented for Qwen3-VL. "
            "Use get_logits_end_to_end or generate_end_to_end instead."
        )
    
    def get_logits_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b seq_len n_tokens"]:
        """Given input embeddings, return per-position logits."""
        # Qwen3 is decoder-only, so we combine embeddings with decoder inputs
        if decoder_input_ids is not None:
            decoder_embeds = self.get_token_embeddings(decoder_input_ids)
            input_embeddings = torch.cat([input_embeddings, decoder_embeds], dim=1)
            if attention_mask is not None and decoder_attention_mask is not None:
                attention_mask = torch.cat([attention_mask, decoder_attention_mask], dim=1)
        
        outputs = self.model.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        
        if decoder_input_ids is not None:
            # Return only the logits corresponding to decoder positions
            return logits[:, -decoder_input_ids.shape[1]:, :]
        return logits
    
    def get_logits_end_to_end(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Int64[Tensor, "b src_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b tgt_seq_len n_tokens"]:
        """Given input tokens and pixel values, return per-position logits."""
        # For Qwen3-VL, we need to use the processor to properly format inputs
        # Combine tokens and decoder_input_ids if provided
        if decoder_input_ids is not None:
            # Strip BOS from decoder if present
            input_ids = torch.cat([tokens, decoder_input_ids[:, 1:]], dim=1)
            if token_attention_mask is not None and decoder_attention_mask is not None:
                attention_mask = torch.cat([token_attention_mask, decoder_attention_mask[:, 1:]], dim=1)
            else:
                attention_mask = None
        else:
            input_ids = tokens
            attention_mask = token_attention_mask
        
        if pixel_values.shape[0] == 1 and input_ids.shape[0] > 1:
            pixel_values = repeat(
                pixel_values, "() c h w -> b c h w", b=input_ids.shape[0]
            )
        
        # Qwen3-VL expects specific input format
        forward_kwargs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        
        # Add image_grid_thw if available (required for Qwen3-VL's dynamic resolution)
        if self._last_image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = self._last_image_grid_thw
        
        outputs = self.model(**forward_kwargs)
        
        if decoder_input_ids is not None:
            return outputs.logits[:, -(decoder_input_ids.shape[1]):, :]
        return outputs.logits
    
    def generate_end_to_end(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Int64[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b new_seq_len n_tokens"]:
        """Given input tokens and pixel values, return generated output tokens."""
        if pixel_values.shape[0] == 1 and tokens.shape[0] > 1:
            pixel_values = repeat(
                pixel_values, "() c h w -> b c h w", b=tokens.shape[0]
            )
        
        # Generate using the model
        generate_kwargs = {
            "pixel_values": pixel_values,
            "input_ids": tokens,
            "attention_mask": token_attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        
        # Add image_grid_thw if available (required for Qwen3-VL's dynamic resolution)
        if self._last_image_grid_thw is not None:
            generate_kwargs["image_grid_thw"] = self._last_image_grid_thw
        
        generated_ids = self.model.generate(**generate_kwargs)
        
        # Trim the input tokens and return only generated tokens
        # Add BOS token at the beginning
        gen_ids = generated_ids[:, tokens.shape[1]:]
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.processor.tokenizer.eos_token_id
        
        return torch.cat([
            torch.full((gen_ids.shape[0], 1), bos_id, device=gen_ids.device),
            gen_ids
        ], dim=1)
    
    def generate_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b new_seq_len n_tokens"]:
        """Given input embeddings, return generated output tokens."""
        generated_ids = self.model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # Add BOS token at the beginning
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.processor.tokenizer.eos_token_id
        
        return torch.cat([
            torch.full((generated_ids.shape[0], 1), bos_id, device=generated_ids.device),
            generated_ids
        ], dim=1)
    
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            # Use EOS as padding if pad token not available
            pad_id = self.processor.tokenizer.eos_token_id
        return pad_id
    
    def loss(
        self,
        logits: Float[Tensor, "b seq_len n_toks"],
        label_toks: Int64[Tensor, "b seq_len"],
        padding_tok: Optional[int] = None,
    ) -> Float[Tensor, ""]:
        """Returns masked language modelling loss."""
        if padding_tok is None:
            padding_tok = self.pad_token_id()
        
        labels = label_toks.to(logits.device)
        logits = logits[:, -labels.size(1):, :]
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=padding_tok)
        vocab_size = self.model.config.vocab_size
        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )
        return loss
    
    @classmethod
    def load_model(
        cls: Type["Qwen3Lens"],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> "Qwen3Lens":
        """Load model and processor."""
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
        
        # Load model
        model = load_model_with_cache(
            model_fn=lambda: Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                device_map="auto",
                cache_dir=CACHE_DIR,
            ).eval(),
            model_id_components=(Path(model_path), model_dtype),
            cache_dir=CACHE_DIR
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=CACHE_DIR)
        
        if not requires_grad:
            model.requires_grad_(False)
        
        return cls(model, processor, model_dtype)


class Qwen3VL8BInstruct(Qwen3Lens):
    """Convenience class for Qwen3-VL-8B-Instruct model"""
    
    @classmethod
    def load_model(
        cls,
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> "Qwen3VL8BInstruct":
        return super().load_model(model_dtype=model_dtype, requires_grad=requires_grad)
