# Motion Correction

Motion correction estimates the translation needed to align every frame in a
movie to a stable reference image. The current implementation uses FFT-based
cross-correlation with an upsampled DFT refinement, following Guizar-Sicairos,
Thurman, and Fienup (2008). The registration error metric follows Fienup
(1997).

The workflow starts with `find_similar_frames`, which identifies stable frames
and averages the top reference frames. `estimate_image_shift` estimates the
row and column shift between one frame and the reference. `estimate_motion_vectors`
applies that shift estimation across a movie, and `apply_motion_correction`
uses those vectors to return a corrected movie.

`WidefieldAnalysis.motion_correction` applies this generic movie correction to
each loaded widefield trial. It updates the trial imaging stack in memory and
stores the motion vectors and reference images on the analysis object. It does
not write processed files, reports, or metadata.

The `integer` shift method rounds motion vectors and uses `roll`, which is fast
and does not interpolate. The `fourier` shift method applies subpixel Fourier
phase shifts, which is slower but preserves subpixel corrections.

References:

- Guizar-Sicairos M, Thurman ST, Fienup JR. Efficient subpixel image
  registration algorithms. Optics Letters. 2008.
- Fienup JR. Invariant error metrics for image reconstruction. Applied Optics.
  1997.
