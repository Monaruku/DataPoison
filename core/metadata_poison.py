"""Metadata manipulation and invisible watermark embedding.

Operations:
  - Strip original EXIF data (GPS, camera info, timestamps)
  - Inject misleading EXIF fields (fake software tag, copyright warning)
  - Embed invisible watermark in LSB of pixel values

The LSB watermark is inherently invisible — it changes only the least
significant bit of each color channel (max change of 1/255 per channel).
"""
import io
import struct
from typing import Optional

import numpy as np
from PIL import Image

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


# Watermark message to embed
WATERMARK_MESSAGE = "DATAPOISON"


def _message_to_bits(message: str) -> list:
    """Convert a string to a flat list of bits (MSB first per byte)."""
    bits = []
    for ch in message:
        byte = ord(ch)
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _embed_lsb_watermark(
    image_array: np.ndarray,
    message: str = WATERMARK_MESSAGE,
    depth: int = 1,
) -> np.ndarray:
    """
    Embed a text message into the least-significant bits of pixel values.

    Args:
        image_array: (H, W, 3) uint8 array.
        message: Text to embed.
        depth: Number of LSBs to use (1 = bit 0 only, 2 = bits 0 and 1).
               Depth 1: max change 1/255 per channel (imperceptible).
               Depth 2: max change 3/255 per channel (still invisible).

    Returns:
        Modified (H, W, 3) uint8 array with embedded watermark.
    """
    result = image_array.copy()
    flat = result.reshape(-1)

    bits = _message_to_bits(message)

    # Repeat message to fill available space
    repeats = max(1, len(flat) // (len(bits) * depth))
    all_bits = bits * repeats

    if depth == 1:
        # Modify only bit 0 (LSB)
        for i, bit in enumerate(all_bits):
            if i >= len(flat):
                break
            flat[i] = (flat[i] & 0xFE) | bit
    elif depth == 2:
        # Modify bits 0 and 1 alternately
        bit_idx = 0
        for i in range(0, len(flat), 2):
            if bit_idx >= len(all_bits):
                break
            # Bit 0 of current pixel
            flat[i] = (flat[i] & 0xFE) | all_bits[bit_idx]
            bit_idx += 1
            if bit_idx >= len(all_bits) or i + 1 >= len(flat):
                break
            # Bit 1 of next pixel (shifted into position 1)
            flat[i + 1] = (flat[i + 1] & 0xFD) | (all_bits[bit_idx] << 1)
            bit_idx += 1

    return result.reshape(image_array.shape)


def apply_metadata_poison(
    pil_image: Image.Image,
    strip_exif: bool = True,
    embed_watermark: bool = True,
    watermark_depth: int = 1,
    output_format: str = "png",
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Apply metadata poisoning to a PIL image.

    Args:
        pil_image: Input PIL Image.
        strip_exif: Remove original EXIF data.
        embed_watermark: Embed invisible LSB watermark.
        watermark_depth: Number of LSB bits to use (1 or 2).
        output_format: Output format (png/jpg/webp). EXIF only works with JPEG.
        save_path: If provided, save with modified EXIF after watermarking.

    Returns:
        Modified PIL Image with embedded watermark.
    """
    result = pil_image.copy()

    # Step 1: Embed invisible watermark in pixel data
    if embed_watermark:
        if result.mode != "RGB":
            result = result.convert("RGB")
        img_array = np.array(result, dtype=np.uint8)
        img_array = _embed_lsb_watermark(img_array, WATERMARK_MESSAGE, watermark_depth)
        result = Image.fromarray(img_array, mode="RGB")

    # Step 2: Handle EXIF (only for JPEG output)
    if HAS_PIEXIF and save_path and output_format.lower() in ("jpg", "jpeg"):
        # Save first, then modify EXIF in-place
        result.save(save_path, format="JPEG", quality=95)

        try:
            if strip_exif:
                piexif.remove(save_path)

            # Build new EXIF data
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Software: b"DataPoison v1.0",
                    piexif.ImageIFD.Copyright: b"Protected - Do not use for AI training",
                    piexif.ImageIFD.Artist: b"Protected by DataPoison",
                },
                "Exif": {},
                "GPS": {},
                "1st": {},
                "thumbnail": None,
            }

            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, save_path)
        except Exception:
            pass  # EXIF modification is best-effort

    return result


def verify_watermark(image_array: np.ndarray, message: str = WATERMARK_MESSAGE) -> bool:
    """
    Verify that a watermark message is present in the LSB of pixel data.
    Returns True if the message is found.
    """
    flat = image_array.reshape(-1)
    expected_bits = _message_to_bits(message)

    # Extract LSBs
    extracted_bits = []
    for i in range(min(len(expected_bits), len(flat))):
        extracted_bits.append(flat[i] & 1)

    # Check if first copy of message matches
    matches = sum(1 for a, b in zip(expected_bits, extracted_bits) if a == b)
    return matches / len(expected_bits) > 0.9  # 90% match threshold
