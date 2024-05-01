# ir_pattern
Generate IR Pattern to gray image.
This code was written in python3.

# 1. Setup
### 1. Dependencies
```bash
pip install -r requirements.txt
```

# 2. Running example code
### 1. apply_dot_depth.py
Add an ir dot pattern to two clean images. Then, a depth image is created using the stereo bm algorithm.
```bash
python3 scripts/apply_dot_depth.py
```
### 2. stereo_bm_example.py
Using the stereo bm algorithm, a depth image is created from the two IR pattern images obtained from the actual D415.
```bash
python3 scripts/stereo_bm_example.py
```
