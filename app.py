from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np

DEBUG_MODE = os.environ.get("DEBUG_MODE", "1") == "1"

# Per-crop contour area limits (in pixels)
# These are starting points – you can tune each crop with a test image.
CROP_AREA_LIMITS = {
    "wheat":   (200, 10000),   # larger, elongated grains
    "barley":  (520, 9000),   # similar to wheat, slightly smaller
    "lentils": (15, 3000),    # tuned from your test image
    "canola":  (60, 2500),     # small, round seed
}

DEFAULT_CROP = "lentils"

# Minimum circularity per crop (0–1). 0 = no circularity filtering.
# Round seeds = higher values, elongated grains = low/zero.
CROP_MIN_CIRCULARITY = {
    "wheat":   0.05,   # allow elongated shapes
    "barley":  0.02,
    "lentils": 0.0,   # fairly round
    "canola":  0.3,   # very round, small
}

app = Flask(__name__)

# CHANGE THIS to something private before you give anyone access
app.secret_key = "CHANGE_THIS_TO_SOMETHING_RANDOM"

# Simple access code for Option C
ACCESS_CODE = "westcoast"  # change this to whatever you want to give clients

def count_seeds(image_path: str, crop: str) -> int:
    """
    Count seeds using adaptive thresholding + contour area + circularity.
    Uses per-crop presets for area and circularity.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0

    # Get crop-specific limits
    min_area, max_area = CROP_AREA_LIMITS.get(crop, CROP_AREA_LIMITS[DEFAULT_CROP])
    min_circ = CROP_MIN_CIRCULARITY.get(crop, 0.0)

    # Grayscale + blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Single adaptive-threshold setting for all crops
    C_value = 10

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        C_value,
    )

    # Morphological clean-up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    seed_count = 0
    debug = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)

        if perim == 0:
            circ = 0.0
        else:
            circ = (4 * np.pi * area) / (perim * perim)

        if (min_area <= area <= max_area) and (circ >= min_circ):
            seed_count += 1
            cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 2)  # counted
        else:
            cv2.drawContours(debug, [cnt], -1, (0, 0, 255), 1)  # rejected

    os.makedirs("uploads", exist_ok=True)
    if DEBUG_MODE:
    os.makedirs("uploads", exist_ok=True)
    cv2.imwrite(os.path.join("uploads", "debug_last.jpg"), debug)

    return int(seed_count)


def is_authenticated() -> bool:
    return bool(session.get("authenticated"))


@app.route("/login", methods=["GET", "POST"])
def login():
    # Already logged in → go straight to main page
    if is_authenticated():
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        code = request.form.get("access_code", "").strip()
        if code == ACCESS_CODE:
            session["authenticated"] = True
            # Reset counts when logging in
            session["seed_counts"] = []
            return redirect(url_for("index"))
        else:
            error = "Incorrect access code."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def index():
    # Require login
    if not is_authenticated():
        return redirect(url_for("login"))

    # Seed counts across multiple images
    if "seed_counts" not in session:
        session["seed_counts"] = []

    # Current crop for counting
    if "current_crop" not in session:
        session["current_crop"] = DEFAULT_CROP

    error = None
    last_seed_count = None

    if request.method == "POST":
        # Update crop choice if provided
        form_crop = request.form.get("crop")
        if form_crop in CROP_AREA_LIMITS:
            session["current_crop"] = form_crop

        # Clear counts
        if "reset" in request.form:
            session["seed_counts"] = []
        else:
            # Normal image upload
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "Please upload an image."
            else:
                filename = secure_filename(file.filename)
                os.makedirs("uploads", exist_ok=True)
                upload_path = os.path.join("uploads", filename)
                file.save(upload_path)

                try:
                    current_crop = session.get("current_crop", DEFAULT_CROP)
                    seed_count = count_seeds(upload_path, current_crop)
                    last_seed_count = seed_count
                    session["seed_counts"].append(int(seed_count))
                except Exception as e:
                    error = f"Error processing image: {e}"

    total_seeds = sum(session.get("seed_counts", []))
    current_crop = session.get("current_crop", DEFAULT_CROP)

    return render_template(
        "index.html",
        seed_count=last_seed_count,
        total_seeds=total_seeds,
        current_crop=current_crop,
        error=error,
    )


# PWA bits: manifest + service worker (safe to leave even if not fully used yet)
@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")


@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")


if __name__ == "__main__":
    # For local testing
    app.run(debug=True)
