"""2-bit log-scale quantization for non-matrix parameter second moments.

Per 128-element block:
  1. Compute log(v_min) and log(v_max)
  2. Define 4 evenly-spaced buckets on log scale
  3. Quantize each element to nearest bucket index (2 bits)
  4. Store: 2 bits/element + 2 fp16 scalars (min, max) per block = 2.25 bits/param

Log-scale chosen because v is strictly positive and spans orders of magnitude.
Adam's update rule (dividing by √v) is most sensitive to small v values;
log-scale spacing allocates more resolution there.
"""
