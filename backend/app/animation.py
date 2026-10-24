from io import BytesIO

from PIL import Image

# GIF는 256색만 쓸 수 있어 사진에서는 색이 뭉개지고, 프레임이 많으면 용량이 급격히
# 커진다. 블로그·SNS에 올릴 수 있는 크기를 유지하려고 폭과 프레임 수를 제한한다.
GIF_MAX_WIDTH = 800
# 전→후로 넘어가는 중간 프레임 수 (많을수록 부드럽지만 용량이 커진다)
GIF_FADE_FRAMES = 8
# 전/후 사진을 그대로 보여주는 시간과, 중간 프레임 한 장당 시간 (밀리초)
GIF_HOLD_MS = 1200
GIF_FADE_MS = 80


def build_crossfade_gif(images: list[Image.Image]) -> bytes:
    """전 → (중간 →) 후 사진으로 차례로 스르륵 넘어가는 GIF를 만든다.

    GIF는 자동으로 반복 재생되므로, 마지막 사진에서 다시 전 사진으로 되돌아오는 구간까지
    넣어야 이어붙는 지점이 튀지 않는다."""
    images = [image.convert("RGB") for image in images]
    first = images[0]

    # 파이프라인 결과는 모두 크기가 같지만, 혹시 다르면 맞춰준다.
    images = [first] + [
        image if image.size == first.size else image.resize(first.size, Image.LANCZOS)
        for image in images[1:]
    ]

    if first.width > GIF_MAX_WIDTH:
        height = round(first.height * GIF_MAX_WIDTH / first.width)
        images = [image.resize((GIF_MAX_WIDTH, height), Image.LANCZOS) for image in images]

    frames = [images[0]]
    durations = [GIF_HOLD_MS]

    for current, following in zip(images, images[1:]):
        for step in range(1, GIF_FADE_FRAMES + 1):
            frames.append(Image.blend(current, following, step / (GIF_FADE_FRAMES + 1)))
            durations.append(GIF_FADE_MS)
        frames.append(following)
        durations.append(GIF_HOLD_MS)

    # 되돌아오는 구간 (반복 재생 시 매끄럽게 이어지도록). 중간을 거치지 않고 마지막에서
    # 전으로 바로 돌아간다 — 거꾸로 거쳐 가면 시간이 거꾸로 흐르는 것처럼 보인다.
    for step in range(GIF_FADE_FRAMES, 0, -1):
        frames.append(Image.blend(images[0], images[-1], step / (GIF_FADE_FRAMES + 1)))
        durations.append(GIF_FADE_MS)

    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return buffer.getvalue()
