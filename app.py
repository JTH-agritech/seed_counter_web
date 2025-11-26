import os
from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Make sure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def count_seeds(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0

    # Resize to keep things manageable and consistent
    h, w = img.shape[:2]
    max_dim = 800
    scale = max_dim / max(h, w)
    if scale < 1.0:  # only shrink, don't enlarge
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Grayscale + blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try both "seeds darker" and "seeds lighter"
    _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th     = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)

    inv_ratio = np.mean(th_inv == 255)
    ratio     = np.mean(th == 255)

    # We roughly expect seeds to cover some fraction of the image.
    target = 0.25
    if abs(inv_ratio - target) < abs(ratio - target):
        mask = th_inv
    else:
        mask = th

    # Clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return 0  # only background found

    # stats[0] = background, ignore it
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)

    # Drop very tiny specks (noise)
    min_area = 15  # tweak this if needed
    areas = areas[areas >= min_area]
    if areas.size == 0:
        return 0

    # Typical single-seed area
    median_area = float(np.median(areas))

    # Overall seed area
    total_area = float(np.sum(areas))

    # Raw estimate = total area / typical seed area
    raw_estimate = total_area / median_area

    # Fudge factor: <1.0 to bring high counts down, >1.0 to push low counts up
    fudge_factor = 0.37  # start with 0.8 since you're currently getting too many

    seeds = int(round(raw_estimate * fudge_factor))
    return max(seeds, 0)


@app.route("/", methods=["GET", "POST"])
def index():
    seed_count = None
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", seed_count=None, error="No file part")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", seed_count=None, error="No selected file")

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Count seeds
        seed_count = count_seeds(filepath)

    return render_template("index.html", seed_count=seed_count, error=None)


if __name__ == "__main__":
    app.run(debug=True)
