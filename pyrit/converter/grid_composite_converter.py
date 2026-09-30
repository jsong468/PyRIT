# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import logging
import random
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image, ImageFont
from PIL.ImageFont import FreeTypeFont

from pyrit.converter.base_image_text_converter import _BaseImageTextConverter
from pyrit.converter.converter import ConverterResult
from pyrit.memory import data_serializer_factory
from pyrit.models import ComponentIdentifier, PromptDataType

logger = logging.getLogger(__name__)


class GridCompositeConverter(_BaseImageTextConverter):
    """
    Renders the prompt text into a composite grid where one cell carries the text
    payload and the remaining cells are innocuous images.

    The grid defaults to 2x2 (four-quadrant split) but any ``grid_size``
    with at least two cells is supported.

    Selection of the payload cell and the innocuous images is seeded from
    ``random_seed`` so a given configuration always produces the same composite,
    keeping the converter's identity (and any downstream technique evaluation
    hash) stable.

    Reference: DMN (ACL 2026) quadrant-splitting; "Beyond Visual Safety"
    (arXiv:2601.15698) neutralized visual splicing.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("image_path",)

    _TEXT_MARGIN = 10

    def __init__(
        self,
        *,
        innocuous_images: Sequence[Path | str],
        grid_size: tuple[int, int] = (2, 2),
        tile_size: tuple[int, int] = (400, 400),
        payload_position: int | None = None,
        payload_background: Path | str | None = None,
        font_name: Path | None = None,
        color: tuple[int, int, int] = (0, 0, 0),
        font_size: int = 20,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize the converter with the innocuous image bank and layout parameters.

        Args:
            innocuous_images (Sequence[Path | str]): Bank of benign image paths (or Azure Blob
                URLs) to fill the non-payload cells. Must contain at least ``rows * cols - 1``
                entries so every non-payload cell can be filled without repetition.
            grid_size (tuple[int, int]): Grid layout as (rows, cols). Must contain at least two
                cells. Defaults to (2, 2).
            tile_size (tuple[int, int]): Size of each cell as (width, height) in pixels; every
                image is resized to this and the payload text is rendered within it.
                Defaults to (400, 400).
            payload_position (int | None): Row-major index (0-based) of the cell that carries the
                objective text. When None (default), a cell is chosen deterministically from
                ``random_seed``.
            payload_background (Path | str | None): Optional background image for the payload cell.
                When None (default), the text is rendered on a white cell.
            font_name (Path | None): Path of the font to use. Must be a TrueType font (.ttf).
                Defaults to None, which uses Pillow's built-in default font.
            color (tuple[int, int, int]): Text color as RGB values. Defaults to (0, 0, 0).
            font_size (int): Font size for the payload text. Defaults to 20.
            random_seed (int): Seed controlling the payload cell and innocuous-image selection.
                Kept in the identifier so identical configurations produce identical composites.
                Defaults to 42.

        Raises:
            ValueError: If ``grid_size`` or ``tile_size`` are not two positive integers, the grid
                has fewer than two cells, ``innocuous_images`` is empty or too small,
                ``payload_position`` is out of range, ``font_name`` is not a ``.ttf`` file, or
                ``font_size`` is not positive.
        """
        if len(grid_size) != 2 or grid_size[0] < 1 or grid_size[1] < 1:
            raise ValueError("grid_size must be a tuple of two positive integers (rows, cols)")
        rows, cols = grid_size
        num_cells = rows * cols
        if num_cells < 2:
            raise ValueError("grid_size must describe at least two cells")
        if len(tile_size) != 2 or tile_size[0] < 1 or tile_size[1] < 1:
            raise ValueError("tile_size must be a tuple of two positive integers (width, height)")
        if not innocuous_images:
            raise ValueError("Please provide a non-empty innocuous_images bank")
        num_innocuous = num_cells - 1
        if len(innocuous_images) < num_innocuous:
            raise ValueError(
                f"innocuous_images must contain at least {num_innocuous} image(s) to fill the "
                f"non-payload cells of a {rows}x{cols} grid; got {len(innocuous_images)}"
            )
        if payload_position is not None and not 0 <= payload_position < num_cells:
            raise ValueError(f"payload_position must be in [0, {num_cells}); got {payload_position}")
        if font_name is not None and Path(font_name).suffix.lower() != ".ttf":
            raise ValueError("The specified font must be a TrueType font with a .ttf extension")
        if font_size < 1:
            raise ValueError("font_size must be greater than 0")

        self._innocuous_images = [str(image) for image in innocuous_images]
        self._grid_size = grid_size
        self._tile_size = tile_size
        self._payload_background = str(payload_background) if payload_background is not None else None
        self._font_name = str(font_name) if font_name is not None else None
        self._color = color
        self._font_size = font_size
        self._random_seed = random_seed
        self._font = self._load_font()

        # Resolve the payload cell and the innocuous subset once, deterministically, so every
        # call with this configuration produces an identical composite.
        rng = random.Random(random_seed)
        self._payload_index = payload_position if payload_position is not None else rng.randrange(num_cells)
        self._selected_innocuous = rng.sample(self._innocuous_images, num_innocuous)

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the converter identifier with layout and text parameters.

        Returns:
            ComponentIdentifier: The identifier for this converter.
        """
        return self._create_identifier(
            params={
                "innocuous_images": sorted(self._innocuous_images),
                "grid_size": self._grid_size,
                "tile_size": self._tile_size,
                "payload_index": self._payload_index,
                "payload_background": self._payload_background,
                "font_name": self._font_name,
                "color": self._color,
                "font_size": self._font_size,
                "random_seed": self._random_seed,
            }
        )

    def _load_font(self) -> FreeTypeFont:
        """
        Load the font at ``self._font_size``.

        Returns:
            FreeTypeFont: The loaded font object. Falls back to Pillow's built-in default font on error.
        """
        if self._font_name is None:
            return cast("FreeTypeFont", ImageFont.load_default(size=self._font_size))
        try:
            return ImageFont.truetype(self._font_name, self._font_size)
        except OSError:
            logger.warning(f"Cannot open font resource: {self._font_name}. Using Pillow built-in default font.")
            return cast("FreeTypeFont", ImageFont.load_default(size=self._font_size))

    @staticmethod
    async def _read_image_async(path: str) -> Image.Image:
        """
        Read an image (local path or Azure Blob URL) and return it decoded.

        Args:
            path (str): The image path or URL.

        Returns:
            Image.Image: The decoded image.
        """
        serializer = data_serializer_factory(category="prompt-memory-entries", value=path, data_type="image_path")
        image_bytes = await serializer.read_data_async()
        return Image.open(BytesIO(image_bytes))

    def _fit_tile(self, image: Image.Image) -> Image.Image:
        """
        Resize an image to the configured tile size.

        Args:
            image (Image.Image): The image to resize.

        Returns:
            Image.Image: The resized RGB image.
        """
        return image.convert("RGB").resize(self._tile_size, Image.Resampling.LANCZOS)

    def _build_payload_tile(self, *, text: str, background: Image.Image | None) -> Image.Image:
        """
        Render the objective text onto the payload cell.

        Args:
            text (str): The objective text to render.
            background (Image.Image | None): Optional background image for the payload cell.
                When None, a white cell is used.

        Returns:
            Image.Image: The rendered payload cell at tile size.
        """
        tile_width, tile_height = self._tile_size
        if background is not None:
            base = self._fit_tile(background)
        else:
            base = Image.new("RGB", self._tile_size, (255, 255, 255))
        bounding_box = (
            self._TEXT_MARGIN,
            self._TEXT_MARGIN,
            tile_width - self._TEXT_MARGIN,
            tile_height - self._TEXT_MARGIN,
        )
        return self._render_text_on_image(
            image=base,
            text=text,
            font=self._font,
            color=self._color,
            bounding_box=bounding_box,
            center_text=True,
        )

    def _compose(self, *, payload_tile: Image.Image, innocuous_tiles: list[Image.Image]) -> Image.Image:
        """
        Paste the payload and innocuous tiles into the grid in row-major order.

        Args:
            payload_tile (Image.Image): The rendered payload cell.
            innocuous_tiles (list[Image.Image]): The innocuous cells, in selection order.

        Returns:
            Image.Image: The composed grid image.
        """
        rows, cols = self._grid_size
        tile_width, tile_height = self._tile_size
        canvas = Image.new("RGB", (cols * tile_width, rows * tile_height), (255, 255, 255))

        innocuous_iter = iter(innocuous_tiles)
        for index in range(rows * cols):
            row, col = divmod(index, cols)
            tile = payload_tile if index == self._payload_index else next(innocuous_iter)
            canvas.paste(tile, (col * tile_width, row * tile_height))
        return canvas

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Render the prompt text into the payload cell of an innocuous composite grid.

        Args:
            prompt (str): The objective text to render into the payload cell.
            input_type (PromptDataType): The type of input data. Must be "text".

        Returns:
            ConverterResult: The result containing the path to the composite image.

        Raises:
            ValueError: If the input type is not supported or ``prompt`` is empty.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        if not prompt:
            raise ValueError("Please provide valid text value")

        background = await self._read_image_async(self._payload_background) if self._payload_background else None
        payload_tile = self._build_payload_tile(text=prompt, background=background)
        innocuous_tiles = [self._fit_tile(await self._read_image_async(path)) for path in self._selected_innocuous]

        composite = self._compose(payload_tile=payload_tile, innocuous_tiles=innocuous_tiles)

        image_bytes = BytesIO()
        composite.save(image_bytes, format="png")
        image_str = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        output_serializer = data_serializer_factory(
            category="prompt-memory-entries", data_type="image_path", extension="png"
        )
        await output_serializer.save_b64_image_async(data=image_str)
        return ConverterResult(output_text=str(output_serializer.value), output_type="image_path")
