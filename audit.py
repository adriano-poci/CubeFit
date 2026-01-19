import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py
from muse import tri_utils as uu

# -------------------------
# EDIT THESE TWO ARRAYS
# -------------------------
# MGE PSF components: I(R) = sum_k w_k * G(sigma_k)
# sigma_k are Gaussian sigmas (NOT FWHM) in arcsec; weights should sum to 1.
psf_sigmas_arcsec = np.array([0.0218712, 0.0433646, 0.1104374, 0.2106640, 0.5392125], dtype=float)   # <-- replace
psf_weights       = np.array([0.4238906, 0.5045164, 0.0444520, 0.0216465, 0.0054946], dtype=float)   # <-- replace
psf_sigmas_arcsec = np.array([0.2972399], dtype=float)   # <-- replace
psf_weights       = np.array([1.0], dtype=float)   # <-- replace
psf_weights = psf_weights / np.sum(psf_weights)

# -------------------------
# INPUTS YOU ALREADY HAVE
# -------------------------
h5_path = '/data/phys-gal-dynamics/phys2603/CubeFit/FCC170/FCC170_3_12.h5'
vb = '/data/phys-gal-dynamics/phys2603/pxf/FCC170/voronoi_SN100_full.xz'
VB = uu.Load.lzma(vb)

binCounts = VB['binCounts']

# These must already be in your session (as in your plotting code):
# xpix, ypix : arrays mapping bins to sky coords (arcsec)
# pixs       : output image pixel size (arcsec) used by your dbi-like plots
# xmin,xmax,ymin,ymax : plot extents in arcsec
# binNum     : index map that dbi uses; if you don't have it, we create a grid below.

# -------------------------
# Read cubes
# -------------------------
with h5py.File(h5_path, "r") as f:
    data_cube  = np.asarray(f["/DataCube"][...], np.float64)
    model_cube = np.asarray(f["/ModelCube"][...], np.float64)
    mask = np.asarray(f["/Mask"][...], bool)

# -------------------------
# Compute SB vectors (per bin)
# -------------------------
data_sb  = np.sum(data_cube[:, mask],  axis=1) / binCounts
model_sb = np.sum(model_cube[:, mask], axis=1) / binCounts

data_sb  = np.where(np.isfinite(data_sb),  data_sb,  np.nan)
model_sb = np.where(np.isfinite(model_sb), model_sb, np.nan)

# -------------------------
# Rasterize bin values onto a regular grid
# -------------------------
# Build an image grid covering [xmin,xmax]x[ymin,ymax] with pixel size pixs
nx = int(np.ceil((xmax - xmin) / pixs))
ny = int(np.ceil((ymax - ymin) / pixs))

# Grid coordinates (pixel centers)
xg = xmin + (np.arange(nx) + 0.5) * pixs
yg = ymin + (np.arange(ny) + 0.5) * pixs

# Map each bin to nearest grid pixel
xi = np.clip(((np.asarray(xpix) - xmin) / pixs).astype(int), 0, nx - 1)
yi = np.clip(((np.asarray(ypix) - ymin) / pixs).astype(int), 0, ny - 1)

# Accumulate (in case multiple bins land in same pixel; average them)
def rasterize(values):
    img = np.full((ny, nx), np.nan, dtype=float)
    wgt = np.zeros((ny, nx), dtype=float)
    val = np.asarray(values, float)

    good = np.isfinite(val)
    y = yi[good]
    x = xi[good]

    # sum and count
    np.add.at(wgt, (y, x), 1.0)
    if np.any(good):
        tmp = np.zeros((ny, nx), dtype=float)
        np.add.at(tmp, (y, x), val[good])
        img = tmp / np.maximum(wgt, 1.0)
        img[wgt == 0] = np.nan
    return img

data_img  = rasterize(data_sb)
model_img = rasterize(model_sb)

# -------------------------
# PSF convolution via MGE (sum of Gaussian filters)
# -------------------------
# Convert sigmas in arcsec to sigmas in pixels
psf_sigmas_pix = psf_sigmas_arcsec / float(pixs)

# Convolve NaN-masked images properly: convolve values and weights separately
def mge_convolve(img):
    img0 = np.asarray(img, float)
    valid = np.isfinite(img0).astype(float)
    img_filled = np.where(np.isfinite(img0), img0, 0.0)

    out_num = np.zeros_like(img_filled)
    out_den = np.zeros_like(valid)

    for w, sig in zip(psf_weights, psf_sigmas_pix):
        if sig <= 0:
            # delta component
            num = img_filled
            den = valid
        else:
            num = gaussian_filter(img_filled, sig, mode="nearest")
            den = gaussian_filter(valid,     sig, mode="nearest")
        out_num += w * num
        out_den += w * den

    out = out_num / np.maximum(out_den, 1e-30)
    out[out_den < 1e-6] = np.nan
    return out

model_img_psf = mge_convolve(model_img)

# Signed residual in SB units, on the image grid
signed_resid_img = data_img - model_img_psf

# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 10), constrained_layout=True)

# log-scale SB panels
def imshow_log(ax, img, title):
    v = img[np.isfinite(img) & (img > 0)]
    vmin = np.nanpercentile(np.log10(v), 1) if v.size else 0
    vmax = np.nanpercentile(np.log10(v), 99) if v.size else 1
    ax.imshow(np.log10(img), origin="lower",
              extent=(xmin, xmax, ymin, ymax),
              vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)

imshow_log(axes[0], data_img, "Data SB (log10)")
imshow_log(axes[1], model_img_psf, "Model SB after MGE PSF (log10)")

# signed residual (linear, symmetric limits)
r = signed_resid_img[np.isfinite(signed_resid_img)]
vlim = np.nanpercentile(np.abs(r), 99) if r.size else 1.0
axes[2].imshow(signed_resid_img, origin="lower",
               extent=(xmin, xmax, ymin, ymax),
               vmin=-vlim, vmax=vlim, aspect="equal")
axes[2].set_title(r"Signed residual SB: Data - PSF(Model)")

for ax in axes:
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")

out_png = "psf_convolved_sb_diagnostics.png"
plt.savefig(out_png, dpi=150)
plt.close(fig)

print("Wrote:", out_png)
print("PSF sigmas (arcsec):", psf_sigmas_arcsec)
print("PSF weights:", psf_weights)
