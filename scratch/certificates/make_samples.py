"""Synthetic calibration certificates with known values, as PDF text and scans."""
import json, math, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

out = Path(sys.argv[1])
K = (3.52886e-05, -3.95205e-09, -1.01226e-13, 1.00869e-17)
def dist_um(r): return (r*K[0] + r**3*K[1] + r**5*K[2] + r**7*K[3]) * 1000
F = 153.692
FID = {"bottom_left": (-106.004, -106.002), "top_right": (106.001, 105.998),
       "top_left": (-105.997, 106.003), "bottom_right": (106.003, -105.999),
       "left_middle": (-112.998, 0.004), "right_middle": (113.002, -0.003),
       "top_middle": (0.002, 113.001), "bottom_middle": (-0.001, -112.996)}
radii = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 148]
random.seed(4)

# Layout A: US survey style, lines as (x_pt, text)
A = [(60, "REPORT OF CALIBRATION"), (60, "of Aerial Mapping Camera"), None,
     (60, "Camera type: Wild RC30          Camera serial no.: 5361"),
     (60, "Lens type: Wild 15/4 UAG-S      Lens serial no.: 13352"),
     (60, "Date of calibration: 12 March 2019"), None,
     (60, "Calibrated focal length: 153.692 mm"),
     (60, "Principal point of autocollimation (PPA):  x = -0.008 mm   y = 0.005 mm"),
     (60, "Principal point of symmetry (PPS):  x = -0.005 mm   y = 0.004 mm"), None,
     (60, "Calibrated fiducial coordinates (mm)"),
     [(60, "Fiducial"), (200, "x"), (300, "y")]]
order = ["bottom_left", "top_right", "top_left", "bottom_right", "left_middle", "right_middle", "top_middle", "bottom_middle"]
for i, s in enumerate(order, 1):
    x, y = FID[s]; A.append([(60, str(i)), (180, f"{x:.3f}"), (280, f"{y:.3f}")])
A += [None, (60, "Radial distortion (micrometres)"),
      [(60, "Radius (mm)"), (150, "Diag. 1"), (220, "Diag. 2"), (290, "Diag. 3"), (360, "Diag. 4"), (440, "Mean")]]
for r in radii:
    d = dist_um(r); diags = [d + random.uniform(-0.6, 0.6) for _ in range(3)]; diags.append(4*d - sum(diags))
    A.append([(70, f"{r}")] + [(150 + 70*i, f"{v:.1f}") for i, v in enumerate(diags)] + [(440, f"{d:.1f}")])

# Layout B: European style, PPO, transposed fiducials, field angle, two columns, no mean
B = [(60, "Kalibrierungsprotokoll / Calibration Certificate"), (60, "Zeiss RMK TOP 15   Serial number: 144321"),
     (60, "Lens: Pleogon A3/4"), (60, "Date: 2021-06-04"), None,
     (60, "Calibrated principal distance c = 153,405 mm"),
     (60, "Principal point offset:  xp = -0,012 mm   yp = 0,009 mm"), None,
     (60, "Fiducial marks (mm)"),
     [(60, "No.")] + [(130 + 55*i, str(i+1)) for i in range(8)],
     [(60, "x")] + [(120 + 55*i, f"{FID[s][0]:.3f}".replace('.', ',')) for i, s in enumerate(order)],
     [(60, "y")] + [(120 + 55*i, f"{FID[s][1]:.3f}".replace('.', ',')) for i, s in enumerate(order)],
     None, (60, "Radial distortion in um"),
     [(60, "Field angle (deg)"), (220, "Axis A"), (320, "Axis B")]]
angles = [5, 10, 15, 20, 25, 30, 35, 40, 42]
for a in angles:
    r = 153.405*math.tan(math.radians(a)); d = dist_um(r); e = random.uniform(-0.4, 0.4)
    B.append([(80, f"{a}"), (220, f"{d+e:.1f}".replace('.', ',')), (320, f"{d-e:.1f}".replace('.', ','))])

truth = {"A": {"focal": 153.692, "ppa": [-0.008, 0.005], "pps": [-0.005, 0.004], "fid": FID, "rows": len(radii), "camera": "Wild RC30"},
         "B": {"focal": 153.405, "ppo": [-0.012, 0.009], "fid": FID, "rows": len(angles), "camera": "Zeiss RMK TOP 15"}}
(out / "truth.json").write_text(json.dumps(truth))

def pdf(lines, path):
    ops = []; y = 800
    for line in lines:
        if line is None: y -= 10; continue
        parts = line if isinstance(line, list) else [line]
        for x, t in parts:
            t = t.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
            ops.append(f"BT /F1 10 Tf {x} {y} Td ({t}) Tj ET")
        y -= 15
    stream = "\n".join(ops).encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    data = b"%PDF-1.4\n"; offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(data)); data += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs)+1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    data += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs)+1, xref)
    path.write_bytes(data)

def scan(lines, path, dpi, rough):
    s = dpi / 72; img = Image.new("L", (int(595*s), int(842*s)), 250); d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(10*s)); y = 842 - 800
    for line in lines:
        if line is None: y += 10; continue
        for x, t in (line if isinstance(line, list) else [line]):
            d.text((x*s, (y-8)*s), t, fill=25, font=font)
        y += 15
    if rough:
        img = img.rotate(0.6, fillcolor=250, resample=Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(0.9))
        a = np.asarray(img).astype(float) + np.random.default_rng(1).normal(0, 14, img.size[::-1])
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    img.save(path, quality=80) if path.suffix == ".jpg" else img.save(path)

for name, lines in (("A", A), ("B", B)):
    pdf(lines, out / f"{name}_text.pdf")
    scan(lines, out / f"{name}_scan.png", 300, False)
    scan(lines, out / f"{name}_rough.jpg", 150, True)
print("made", sorted(p.name for p in out.iterdir()))
