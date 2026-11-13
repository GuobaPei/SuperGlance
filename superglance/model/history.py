"""Past front-camera frames: one definition for the training rows, every evaluation and the AlpaSim driver.
- Past frames t - 1.5 / 1.0 / 0.5 s (OFFS, oldest first), then the current frame; "frame: k" before each image, the
  current frame last (frame 3).
- A past frame is shrunk to at most HIST_PIXELS (bicubic, aspect kept) and JPEG q92 encoded; the processor then rounds
  556 x 313 to 544 x 320 (170 visual tokens). Training reads the stored JPEG; an evaluation passes a full-size frame
  through hist_jpeg first.
- The question's camera sentence ONE becomes MANY; nothing else in the text changes."""
import io

from PIL import Image

HIST_PIXELS, QUALITY, OFFS = 174080, 92, (1.5, 1.0, 0.5)
ONE = "the image is its front camera"
MANY = "the images are its front camera 1.5 s, 1.0 s and 0.5 s ago (frames 0-2) and now (frame 3)"


def shrink(image):
    """At most HIST_PIXELS pixels, aspect ratio kept (a smaller image is returned as is)."""
    w, h = image.size
    scale = (HIST_PIXELS / (w * h)) ** 0.5
    return image.resize((max(32, round(w * scale)), max(32, round(h * scale))), Image.BICUBIC) if scale < 1 else image


def hist_jpeg(image):
    """A full-size RGB frame -> the stored past-frame JPEG bytes (shrink, then JPEG q92)."""
    b = io.BytesIO()
    shrink(image.convert("RGB")).save(b, format="JPEG", quality=QUALITY)
    return b.getvalue()


def open_past(p):
    """A past frame for the collator: a stored JPEG path, JPEG bytes, or a PIL image (shrunk if it is larger)."""
    if isinstance(p, Image.Image):
        return shrink(p.convert("RGB"))
    with Image.open(io.BytesIO(p) if isinstance(p, bytes) else p) as image:
        return shrink(image.convert("RGB"))


def content(n_past):
    """The user turn's content before the question: "frame: k" and an image placeholder per frame, the current one last."""
    return [x for k in range(n_past + 1) for x in ({"type": "text", "text": f"frame: {k}"}, {"type": "image"})]


def question(q):
    assert q.count(ONE) == 1, q[:200]
    return q.replace(ONE, MANY)


if __name__ == "__main__":   # self-check
    im = Image.new("RGB", (1920, 1080), (90, 120, 150))
    s = shrink(im); assert s.size == (556, 313) and s.size[0] * s.size[1] <= HIST_PIXELS, s.size
    j = Image.open(io.BytesIO(hist_jpeg(im))); assert j.size == (556, 313) and shrink(j).size == (556, 313)
    assert open_past(hist_jpeg(im)).size == (556, 313) and open_past(im).size == (556, 313)
    c = content(3); assert [x["type"] for x in c] == ["text", "image"] * 4 and c[-2]["text"] == "frame: 3"
    q = "You are driving the ego vehicle; the image is its front camera. Speed 1.0 m/s."
    assert question(q) == "You are driving the ego vehicle; " + MANY + ". Speed 1.0 m/s."
    print("MF_HIST_OK")
