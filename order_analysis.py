import zipfile
import xml.etree.ElementTree as ET
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def load_shared_strings(z):
    strings = []

    if "xl/sharedStrings.xml" not in z.namelist():
        return strings

    for event, elem in ET.iterparse(z.open("xl/sharedStrings.xml"), events=("end",)):
        if elem.tag == NS + "si":
            texts = []
            for t in elem.iter(NS + "t"):
                texts.append(t.text or "")
            strings.append("".join(texts))
            elem.clear()

    return strings


def col_index(cell_ref):
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)

    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64

    return n - 1


def read_xlsx_numeric(path):
    with zipfile.ZipFile(path) as z:
        shared = load_shared_strings(z)

        nrows = 0

        with z.open("xl/worksheets/sheet1.xml") as f:
            for ev, elem in ET.iterparse(f, events=("start",)):
                if elem.tag == NS + "dimension":
                    ref = elem.attrib.get("ref", "")
                    m = re.search(r":([A-Z]+)(\d+)", ref)
                    nrows = int(m.group(2)) - 1 if m else 0
                    break

        arr = np.empty((nrows, 5), dtype=np.float64)
        headers = [None] * 5
        ri = -1

        with z.open("xl/worksheets/sheet1.xml") as f:
            for ev, row in ET.iterparse(f, events=("end",)):
                if row.tag != NS + "row":
                    continue

                rnum = int(row.attrib.get("r", "0"))
                vals = [np.nan] * 5

                for c in row.findall(NS + "c"):
                    ref = c.attrib.get("r", "")
                    j = col_index(ref)

                    if j >= 5:
                        continue

                    typ = c.attrib.get("t")
                    v = c.find(NS + "v")

                    if v is None:
                        continue

                    txt = v.text

                    if rnum == 1:
                        headers[j] = shared[int(txt)] if typ == "s" else txt
                    else:
                        vals[j] = float(txt)

                if rnum > 1:
                    ri += 1
                    arr[ri, :] = vals

                row.clear()

        return headers, arr[:ri + 1]


def angular_resample(time, rpm, signal, samples_per_rev=512):
    mask = (
        np.isfinite(time)
        & np.isfinite(rpm)
        & np.isfinite(signal)
        & (rpm > 0)
    )

    time = time[mask]
    rpm = rpm[mask]
    signal = signal[mask]

    dt = np.diff(time, prepend=time[0])

    dt[0] = np.median(
        np.diff(time[:min(len(time), 10000)])
    )

    omega = 2 * np.pi * rpm / 60.0
    theta = np.cumsum(omega * dt)

    keep = np.r_[True, np.diff(theta) > 0]

    theta = theta[keep]
    signal = signal[keep]
    rpm = rpm[keep]

    dtheta = 2 * np.pi / samples_per_rev

    theta_u = np.arange(theta[0], theta[-1], dtheta)

    x_u = np.interp(theta_u, theta, signal)
    rpm_u = np.interp(theta_u, theta, rpm)

    return theta_u, x_u, rpm_u


def order_map(
    theta_u,
    x_u,
    rpm_u,
    samples_per_rev=512,
    revs_per_block=8,
    overlap=0.75,
    max_order=30
):
    nper = int(samples_per_rev * revs_per_block)
    hop = max(1, int(nper * (1 - overlap)))

    win = np.hanning(nper)
    win_sum = np.sum(win)

    orders = np.fft.rfftfreq(
        nper,
        d=1 / samples_per_rev
    )

    keep = orders <= max_order
    orders = orders[keep]

    specs = []
    rpms = []

    for start in range(0, len(x_u) - nper + 1, hop):
        block = x_u[start:start + nper]
        block = block - np.mean(block)

        X = np.fft.rfft(block * win)

        # PEAK amplitude scaling
        # Eski RMS benzeri scaling:
        # amp = np.sqrt(2) * np.abs(X) / win_sum
        amp = (
            2.0
            * np.abs(X)
            / win_sum
        )

        specs.append(amp[keep])
        rpms.append(np.mean(rpm_u[start:start + nper]))

    return orders, np.asarray(rpms), np.asarray(specs)


def smooth_curve(y, window_length=9, polyorder=2):
    y = np.asarray(y)

    if len(y) < window_length:
        return y

    if window_length % 2 == 0:
        window_length += 1

    if window_length >= len(y):
        window_length = len(y) - 1

    if window_length % 2 == 0:
        window_length -= 1

    if window_length < 5:
        return y

    return savgol_filter(
        y,
        window_length=window_length,
        polyorder=polyorder
    )


def resample_to_rpm_step(rpm, amp, rpm_step=10):
    rpm = np.asarray(rpm)
    amp = np.asarray(amp)

    mask = np.isfinite(rpm) & np.isfinite(amp)

    rpm = rpm[mask]
    amp = amp[mask]

    if len(rpm) < 2:
        return rpm, amp

    sort_idx = np.argsort(rpm)

    rpm = rpm[sort_idx]
    amp = amp[sort_idx]

    rpm_min = np.ceil(rpm[0] / rpm_step) * rpm_step
    rpm_max = np.floor(rpm[-1] / rpm_step) * rpm_step

    if rpm_max <= rpm_min:
        return rpm, amp

    rpm_grid = np.arange(
        rpm_min,
        rpm_max + rpm_step,
        rpm_step
    )

    amp_grid = np.interp(
        rpm_grid,
        rpm,
        amp
    )

    return rpm_grid, amp_grid


def plot_order_map(
    orders,
    rpms,
    spec,
    channel_name="Channel",
    db_reference=1.0
):
    idx = np.argsort(rpms)

    r = rpms[idx]
    s = spec[idx]

    db = 20 * np.log10(
        np.maximum(s, 1e-12) / db_reference
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    im = ax.imshow(
        db,
        aspect="auto",
        origin="lower",
        extent=[
            orders[0],
            orders[-1],
            r[0],
            r[-1]
        ],
        interpolation="nearest",
        cmap="jet"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Amplitude [dB re 1 m/s²]"
    )

    ax.set_xlabel("Order")
    ax.set_ylabel("RPM")
    ax.set_title(f"Order Map - {channel_name}")

    return fig


def extract_order_vs_rpm(
    orders,
    rpms,
    spec,
    target_order=10.0,
    width=0.15,
    rpm_step=10,
    smooth=True
):
    band = (
        (orders >= target_order - width / 2)
        &
        (orders <= target_order + width / 2)
    )

    if not np.any(band):
        order_idx = np.argmin(
            np.abs(orders - target_order)
        )
        amp = spec[:, order_idx]
    else:
        # Band energy integration
        amp = np.sqrt(
            np.sum(
                spec[:, band] ** 2,
                axis=1
            )
        )

    sort_idx = np.argsort(rpms)

    rpm_sorted = rpms[sort_idx]
    amp_sorted = amp[sort_idx]

    if smooth:
        amp_sorted = smooth_curve(
            amp_sorted,
            window_length=9,
            polyorder=2
        )

    rpm_step_sorted, amp_step_sorted = resample_to_rpm_step(
        rpm_sorted,
        amp_sorted,
        rpm_step=rpm_step
    )

    return rpm_step_sorted, amp_step_sorted
