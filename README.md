# LumenEye AI

LumenEye AI is a local desktop computer vision system for custom object identity detection.
It lets you define your own object models from a live camera feed, capture multiple samples,
and verify whether the object currently in view is the same object you registered earlier.

The project is designed for practical identity-aware recognition rather than generic object
classification alone. A model can be created for a specific device, accessory, tool, package,
or any other physical object you want to track and verify in a live scene.

## Core Capabilities

- Local desktop application with a live camera interface
- User-defined object models with multi-sample registration
- Open-vocabulary detection powered by YOLOWorld
- Identity verification using feature matching, appearance scoring, and object embeddings
- Self-face registration and presence signals
- Event timeline and model registry views
- Guided capture workflow for collecting multiple samples quickly

## How It Works

LumenEye AI uses a two-stage pipeline:

1. The detector searches the full frame for candidate objects using open-vocabulary prompts.
2. Each candidate is verified against the saved samples of your registered model before it is accepted as a match.

This makes the system more selective than a detector-only workflow and better suited for
recognizing a specific registered object instead of any item from the same broad category.

## Typical Use Cases

- Register a specific phone, watch, bottle, or product package and verify it in a live scene
- Build custom object identities for demos, prototypes, or computer vision experiments
- Compare visually similar objects and reject near-matches that are not the exact registered model
- Evaluate whether a registered object remains recognizable from different angles and distances

## Desktop Workflow

The main application is a local desktop app. You can:

- Create a new model with a custom name, label, category, and expected color
- Draw a region of interest over the object you want to save
- Freeze the preview while selecting the ROI
- Refresh the live feed if you want to cancel the current ROI
- Add multiple samples to the same model for better identity verification
- Review saved sample thumbnails for the selected model
- Monitor live matches, detector status, and event history

## Installation

```bash
py -3.10 -m pip install -r requirements.txt
py -3.10 -m pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
```

## Run

```bash
py -3.10 app.py
```

## Project Structure

- `app.py`: application entry point and runtime loop
- `desktop/`: desktop dashboard and UI workflow
- `vision/`: detector, object identity logic, and face processing
- `core/`: configuration, utilities, events, and rules
- `tests/`: automated tests for the model pipeline
- `tools/`: auxiliary scripts such as webcam field tests

## Webcam Field Test

You can run a practical webcam evaluation against a registered model:

```bash
py -3.10 tools\webcam_field_test.py --model blue_bic_pen --seconds 20 --show-window
```

The script reports:

- total frames
- matched frames
- match rate
- best match score
- best detector score
- prompts used during the session

This is useful for validating a model before a demo or presentation.

## Notes

- The system runs locally and does not require a cloud inference service for normal use.
- Recognition quality depends on the number and variety of samples you collect for each model.
- Very similar objects may still require more samples or stricter tuning for perfect separation.
