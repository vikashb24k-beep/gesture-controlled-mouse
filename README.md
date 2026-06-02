# Hand Gesture Mouse

Control your laptop mouse using hand gestures through the webcam.

## Requirements

- Python 3.14
- Webcam
- Packages listed in `requirements.txt`
- `hand_landmarker.task` model file in this project folder

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Controls

- Thumb: move cursor
- Thumb + index finger: click
- Thumb + middle finger: scroll up
- Thumb + little finger: scroll down
- `Esc`: quit

## Notes

If the camera does not open, close other apps that may be using the webcam.
PyAutoGUI fail-safe is enabled, so moving the cursor to a screen corner can stop automated mouse control.
