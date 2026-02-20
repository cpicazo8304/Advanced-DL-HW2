from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        # tokenize image
        tokens = self.tokenizer.encode_index(x)  # (B, h, w)
        B, h, w = tokens.shape

        # get all probabilities at once
        logits, _ = self.autoregressive.forward(tokens)  # (B, h, w, n_tokens)
        probs = torch.softmax(logits, dim=-1)  # (B, h, w, n_tokens)

        # arithmetic coding
        low, high = 0.0, 1.0

        for i in range(h * w):
            r, c = i // w, i % w

            # get prob distribution at position i
            p = probs[0, r, c, :]  # (n_tokens,)

            # get cumulative probs (CDF)
            cdf = torch.cumsum(p, dim=0)
            cdf = torch.cat([torch.tensor([0.0]), cdf])  # prepend 0

            # get actual token at position i
            token = tokens[0, r, c].item()

            # narrow interval
            width = high - low
            high = low + width * cdf[token + 1].item()
            low = low + width * cdf[token].item()

        # pick a number in the middle of the final interval
        code = (low + high) / 2.0

        # convert float to bytes
        import struct
        return struct.pack('d', code)  # 'd' = double = 8 bytes
        

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        import struct
    
        # unpack the float
        code = struct.unpack('d', x)[0]
        
        h, w = 150, 100  # need to know the shape ahead of time
        B = 1
        
        tokens = torch.zeros(B, h, w, dtype=torch.long)
        
        low, high = 0.0, 1.0
        
        for i in range(h * w):
            r, c = i // w, i % w
            
            # get probs at position i using tokens generated so far
            logits, _ = self.autoregressive.forward(tokens)
            p = torch.softmax(logits, dim=-1)[0, r, c, :]  # (n_tokens,)
            
            # build CDF
            cdf = torch.cumsum(p, dim=0)
            cdf = torch.cat([torch.tensor([0.0]), cdf])
            
            # normalize code to current interval
            normalized = (code - low) / (high - low)
            
            # find which token's interval the code falls in
            token = (cdf <= normalized).sum().item() - 1
            tokens[0, r, c] = token
            
            # narrow interval same as encoding
            width = high - low
            high = low + width * cdf[token + 1].item()
            low = low + width * cdf[token].item()
        
        # detokenize back to image
        return self.tokenizer.decode_index(tokens)


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
