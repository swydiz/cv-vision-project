def get_position(x1, y1, x2, y2, w, h):

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < w / 3:
        horiz = "слева"
    elif cx < 2 * w / 3:
        horiz = "по центру"
    else:
        horiz = "справа"

    # vertical
    if cy < h / 3:
        vert = "сверху"
    elif cy < 2 * h / 3:
        vert = "в центре"
    else:
        vert = "снизу"

    if horiz == "по центру" and vert == "в центре":
        return "перед вами"

    if horiz == "по центру":
        return vert
    if vert == "в центре":
        return horiz

    return f"{horiz} {vert}"

def build_description(boxes, labels, scores, w, h, classes, conf=0.3):

    objects = []

    for box, label, score in zip(boxes, labels, scores):

        if score < conf:
            continue

        x1, y1, x2, y2 = box

        name = classes[int(label)] if label < len(classes) else "объект"
        position = get_position(x1, y1, x2, y2, w, h)

        objects.append(f"{name} {position}")

    if len(objects) == 0:
        return "Объекты не обнаружены"

    sentences = [f"Я вижу {name} {pos}" for name, pos in [o.split(" ", 1) for o in objects]]

    return " ".join(sentences)