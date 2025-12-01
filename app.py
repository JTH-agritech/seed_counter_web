from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np

from PIL import Image, ExifTags

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

def detect_mobile_image(image_path: str, width: int, height: int) -> bool:
    """
    Detect whether an image likely came from a mobile device.
    """
    max_dim = max(width, height)
    
    # 1. Check resolution
    if max_dim < 2600:
        return True

    # 2. Try EXIF metadata
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        if exif:
            for k, v in ExifTags.TAGS.items():
                if v == "Make":
                    make_key = k
                    break
            camera_make = exif.get(make_key, "").lower()
            if any(x in camera_make for x in ["apple", "iphone", "samsung", "pixel", "huawei"]):
                return True
    except Exception:
        pass

    return False

def count_seeds(image_path: str, crop: str) -> int:
    """
    Count seeds using adaptive thresholding + contour area + circularity.
    Includes automatic mobile/desktop detection and per-crop tuning.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0

    h, w = image.shape[:2]
    max_dim = float(max(h, w))

    # --- AUTO-DETECT MOBILE ---
    is_mobile = detect_mobile_image(image_path, w, h)

    # Reference sizes used for scaling area thresholds
    MOBILE_REF = 2000.0
    DESKTOP_REF = 3500.0

    # --- CROP DEFAULT LIMITS ---
    raw_min, raw_max = CROP_AREA_LIMITS.get(crop, CROP_AREA_LIMITS["wheat"])
    raw_circ = CROP_MIN_CIRCULARITY.get(crop, 0.0)

    # --- MOBILE VS DESKTOP BRANCHES ---
    if is_mobile:
        # Mobile images smaller & noisier
        scale_factor = (max_dim / MOBILE_REF) ** 2

        # strengthen mobile-specific filters
        min_area = raw_min * 1.6 * scale_factor      # 60% more strict
        max_area = raw_max * scale_factor
        min_circ = raw_circ + 0.05                   # slightly rounder
        blur_size = 7                                # heavier blur
    else:
        scale_factor = (max_dim / DESKTOP_REF) ** 2

        min_area = raw_min * scale_factor
        max_area = raw_max * scale_factor
        min_circ = raw_circ
        blur_size = 5

    # --- PROCESSING ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Adaptive threshold
    C_value = 10
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        C_value,
    )

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- CONTOURS ---
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
            cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(debug, [cnt], -1, (0, 0, 255), 1)

    # Save debug image (optional)
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

@app.route("/debug")
def debug_image():
    if not DEBUG_MODE:
        return "Debug mode disabled", 403

    debug_path = os.path.join("uploads", "debug_last.jpg")
    if not os.path.exists(debug_path):
        return "No debug image yet.", 404

    return send_from_directory("uploads", "debug_last.jpg")

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
