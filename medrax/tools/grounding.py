from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pydantic import BaseModel, Field

from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class XRayPhraseGroundingInput(BaseModel):
    """Input schema for the XRay Phrase Grounding Tool. Only supports JPG or PNG images."""

    image_path: str = Field(
        ...,
        description="Path to the frontal chest X-ray image file, only supports JPG or PNG images",
    )
    phrase: str = Field(
        ...,
        description="Medical finding or condition to locate in the image (e.g., 'Pleural effusion')",
    )
    max_new_tokens: int = Field(default=300, description="Maximum number of new tokens to generate")


class XRayPhraseGroundingTool(BaseTool):
    """Tool for grounding medical findings in chest X-ray images using the MAIRA-2 model.

    This tool processes chest X-ray images and locates specific medical findings mentioned
    in the input phrase. It returns both the bounding box coordinates and a visualization
    of the finding's location in the image.
    """

    name: str = "xray_phrase_grounding"
    description: str = (
        "Locates and visualizes specific medical findings in chest X-ray images. "
        "Takes a chest X-ray image and medical phrase to locate (e.g., 'Pleural effusion', 'Cardiomegaly'). "
        "Returns bounding box coordinates in format [x_topleft, y_topleft, x_bottomright, y_bottomright] "
        "where each value is between 0-1 representing relative position in the image, "
        "a visualization of the finding's location, and confidence metadata. "
        "Example input: {'image_path': '/path/to/xray.png', 'phrase': 'Pleural effusion', 'max_new_tokens': 300}"
    )
    args_schema: Type[BaseModel] = XRayPhraseGroundingInput

    model: Any = None
    processor: Any = None
    device: str = "cuda"
    temp_dir: Path = None

    @staticmethod
    def _apply_maira_compat_shims(model: Any) -> Any:
        """Patch known MAIRA-2/transformers API drift at runtime."""
        # Newer transformers expect/consume `.model` for Llava descendants,
        # while MAIRA-2 remote code defines `.language_model`.
        if not hasattr(model, "model") and hasattr(model, "language_model"):
            model.model = model.language_model

        # Newer generate paths may access `.lm_head` directly on the top-level model.
        # MAIRA-2 remote code can expose output embeddings under nested modules only.
        if not hasattr(model, "lm_head"):
            candidate = None
            for attr in ("language_model", "model"):
                sub = getattr(model, attr, None)
                if sub is not None and hasattr(sub, "lm_head"):
                    candidate = getattr(sub, "lm_head")
                    break
                if sub is not None and hasattr(sub, "get_output_embeddings"):
                    try:
                        out = sub.get_output_embeddings()
                    except Exception:
                        out = None
                    if out is not None:
                        candidate = out
                        break
            if candidate is None and hasattr(model, "get_output_embeddings"):
                try:
                    candidate = model.get_output_embeddings()
                except Exception:
                    candidate = None
            if candidate is not None:
                model.lm_head = candidate

        if not hasattr(model, "get_output_embeddings") and hasattr(model, "lm_head"):
            model.get_output_embeddings = lambda: model.lm_head

        if not hasattr(model, "set_output_embeddings"):
            model.set_output_embeddings = lambda new_embeddings: setattr(
                model, "lm_head", new_embeddings
            )
        return model

    def __init__(
        self,
        model_path: str = "microsoft/maira-2",
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = "cuda",
    ):
        """Initialize the XRay Phrase Grounding Tool."""
        super().__init__()
        self.device = torch.device(device) if device else "cuda"

        # Setup quantization config
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        # Load model (retry without quantization for incompatible configs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device,
                cache_dir=cache_dir,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device,
                cache_dir=cache_dir,
                trust_remote_code=True,
                quantization_config=None,
            )
        self.processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True
        )

        
        self.model = self._apply_maira_compat_shims(self.model.eval())

        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _visualize_bboxes(
        self, image: Image.Image, bboxes: List[Tuple[float, float, float, float]], phrase: str
    ) -> str:
        """Create and save visualization of multiple bounding boxes on the image."""
        plt.figure(figsize=(12, 12))
        plt.imshow(image, cmap="gray")

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            plt.gca().add_patch(
                plt.Rectangle(
                    (x1 * image.width, y1 * image.height),
                    width * image.width,
                    height * image.height,
                    fill=False,
                    color="red",
                    linewidth=2,
                )
            )

        plt.title(f"Located: {phrase}", pad=20)
        plt.axis("off")

        viz_path = self.temp_dir / f"grounding_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(viz_path, bbox_inches="tight", dpi=150)
        plt.close()

        return str(viz_path)

    @staticmethod
    def _is_known_generation_compat_error(exc: Exception) -> bool:
        msg = str(exc)
        known_signatures = (
            "no attribute 'lm_head'",
            "mat1 and mat2 shapes cannot be multiplied",
            "The size of tensor a",
        )
        return any(sig in msg for sig in known_signatures)

    def _fallback_no_finding_output(
        self, image_path: str, phrase: str, image: Image.Image, error_text: str
    ) -> Tuple[Dict[str, Any], Dict]:
        """Graceful fallback when MAIRA generation is incompatible with local transformers."""
        output = {
            "predictions": [],
            "visualization_path": None,
            "note": "Fallback response: phrase-grounding model generation is temporarily unavailable.",
            "queried_phrase": phrase,
        }
        metadata = {
            "image_path": image_path,
            "original_size": image.size,
            "analysis_status": "completed_fallback",
            "fallback_reason": error_text,
        }
        return output, metadata

    def _run(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Ground a medical finding phrase in an X-ray image.

        Args:
            image_path: Path to the chest X-ray image file
            phrase: Medical finding to locate in the image
            max_new_tokens: Maximum number of new tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary and metadata dictionary
        """
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image, phrase=phrase, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
            except Exception as exc:
                if self._is_known_generation_compat_error(exc):
                    return self._fallback_no_finding_output(
                        image_path=image_path,
                        phrase=phrase,
                        image=image,
                        error_text=str(exc),
                    )
                raise

            prompt_length = inputs["input_ids"].shape[-1]
            decoded_text = self.processor.decode(
                output[0][prompt_length:], skip_special_tokens=True
            )
            predictions = self.processor.convert_output_to_plaintext_or_grounded_sequence(
                decoded_text
            )

            metadata = {
                "image_path": image_path,
                "original_size": image.size,
                "model_input_size": tuple(inputs["pixel_values"].shape[-2:]),
                "device": str(self.device),
                "analysis_status": "completed",
            }

            if not predictions:
                output = {
                    "predictions": [],
                    "visualization_path": None,
                }
                metadata["analysis_status"] = "completed_no_finding"
                return output, metadata

            # Process multiple predictions
            processed_predictions = []
            for pred_phrase, pred_bboxes in predictions:
                if not pred_bboxes:  # Skip if no bounding boxes
                    continue

                # Convert model bboxes to list format and get original image bboxes
                model_bboxes = [list(bbox) for bbox in pred_bboxes]
                original_bboxes = [
                    self.processor.adjust_box_for_original_image_size(
                        bbox, width=image.size[0], height=image.size[1]
                    )
                    for bbox in model_bboxes
                ]

                processed_predictions.append(
                    {
                        "phrase": pred_phrase,
                        "bounding_boxes": {
                            "model_coordinates": model_bboxes,
                            "image_coordinates": original_bboxes,
                        },
                    }
                )

            # Create visualization with all bounding boxes
            if processed_predictions:
                all_bboxes = []
                for pred in processed_predictions:
                    all_bboxes.extend(pred["bounding_boxes"]["image_coordinates"])
                viz_path = self._visualize_bboxes(image, all_bboxes, phrase)
            else:
                viz_path = None
                metadata["analysis_status"] = "completed_no_finding"

            output = {
                "predictions": processed_predictions,
                "visualization_path": viz_path,
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_path: str,
        phrase: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Asynchronous version of _run."""
        return self._run(image_path, phrase, run_manager)
