"""
fhcpfsk.py — Reference transmitter and direct-sampling receiver for the
FH-CPFSK system described in SPEC.md (matching the MATLAB cpfsk_tx.m).

This is the "answer key" library used throughout the course. Every function
here is deliberately written to be readable and to match the specification
exactly. The notebooks develop these ideas step by step and validate their
own code against this module.

All processing is done at the full simulation rate fs = 8.192 GHz on a
real-valued waveform, exactly as a direct-sampling receiver would see it.
"""

import numpy as np
from scipy import signal

# ----------------------------------------------------------------------------
# System parameters (from SPEC.md / cpfsk_tx.m)
# ----------------------------------------------------------------------------
Rb          = 8e6           # bit rate, 8 Mbps
Tb          = 1.0 / Rb      # bit period, 125 ns
SPS         = 1024          # samples per bit at the simulation rate
FS          = SPS * Rb      # simulation / ADC sample rate, 8.192 GHz
FC0         = 2e9           # hop-set centre, 2 GHz
NCHAN       = 6             # number of channels
HOP_SPACING = 18e6          # channel spacing, 18 MHz
DEV         = Rb / 2        # frequency deviation from carrier, ±4 MHz (h=1)

PREAMBLE = [1, 0] * 4                # 8 bits, alternating 1010...
SFD      = [1, 1, 1, 0, 1]           # Barker-5
HDR_BITS = 3                         # sequence-number width
PAYLOAD_BITS = 64
CRC_BITS = 16
FRAME_BITS  = len(PREAMBLE) + len(SFD) + HDR_BITS + PAYLOAD_BITS + CRC_BITS  # 96
SAMPLES_PER_FRAME = FRAME_BITS * SPS   # 98304 samples per dwell

RAMP_BITS = 0.5             # raised-cosine amplitude ramp at each dwell edge


def hop_set():
    """Return the six channel carrier frequencies (Hz): 1.955 .. 2.045 GHz."""
    return FC0 + (np.arange(1, NCHAN + 1) - (NCHAN + 1) / 2) * HOP_SPACING


# ----------------------------------------------------------------------------
# CRC-16/CCITT-FALSE  (bit-wise: the covered field HDR+PAYLOAD is 67 bits,
# not byte-aligned, so the CRC is computed directly over the bit stream)
# ----------------------------------------------------------------------------
def crc16_ccitt_false(bits):
    """CRC-16/CCITT-FALSE over a bit stream (MSB first).
    poly 0x1021, init 0xFFFF, no reflection, no final XOR."""
    crc = 0xFFFF
    for b in bits:
        msb = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if msb ^ (int(b) & 1):
            crc ^= 0x1021
    return crc


def bits_to_bytes(bits):
    """Pack a list/array of bits (MSB first) into a list of byte values."""
    bits = list(bits)
    assert len(bits) % 8 == 0
    out = []
    for i in range(0, len(bits), 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | int(b)
        out.append(v)
    return out


def int_to_bits(value, nbits):
    """MSB-first bit list of an integer."""
    return [(value >> i) & 1 for i in range(nbits - 1, -1, -1)]


# ----------------------------------------------------------------------------
# Frame assembly
# ----------------------------------------------------------------------------
def build_frame_bits(message, seq):
    """Assemble the 96-bit frame for an 8-character message and sequence number."""
    assert len(message) == 8, "message must be exactly 8 ASCII characters"
    hdr = int_to_bits(seq, HDR_BITS)
    payload = []
    for ch in message:
        payload += int_to_bits(ord(ch), 8)
    crc = crc16_ccitt_false(hdr + payload)          # over HDR+PAYLOAD (67 bits)
    crc_bits = int_to_bits(crc, CRC_BITS)
    frame = PREAMBLE + SFD + hdr + payload + crc_bits
    assert len(frame) == FRAME_BITS
    return frame


# ----------------------------------------------------------------------------
# CPFSK modulation (continuous phase)
# ----------------------------------------------------------------------------
def cpfsk_modulate(bits, fc, fs=FS, sps=SPS, dev=DEV, ramp_bits=RAMP_BITS,
                   phase_tol=0.0, phase_tol_mode="none",
                   bit_tol=0.0, bit_tol_mode="none", rng=None):
    """Continuous-phase FSK. bit 1 -> +dev (upper tone), bit 0 -> -dev (lower tone).

    ramp_bits : raised-cosine amplitude ramp at each dwell edge, in bits
                (splatter control), matching cpfsk_tx.m; 0 = hard on/off.

    Optional transmitter impairments, all defaulting to OFF (ideal), mirroring
    cpfsk_tx.m:
      phase_tol : fractional error on the +/-180 deg phase accumulated per bit,
                  i.e. an effective modulation index h = 1 +/- phase_tol.
      bit_tol   : fractional error on the symbol-clock period.
      *_mode    : 'none' | 'burst'/'hop' (one draw for this call) |
                  'bit' (redrawn every bit).
    """
    bits = list(bits)
    nbits = len(bits)
    signs = np.array([1.0 if b else -1.0 for b in bits])

    if phase_tol == 0.0 and bit_tol == 0.0:
        # Ideal phase/timing path.
        f_inst = fc + dev * np.repeat(signs, sps)      # instantaneous frequency per sample
        phase = 2 * np.pi * np.cumsum(f_inst) / fs     # continuous phase (integral of freq)
        sig = np.cos(phase)
    else:
        # General path with impairments.
        if rng is None:
            rng = np.random.default_rng()

        def _draw(tol, mode):
            if mode == "none" or tol == 0.0:
                return np.zeros(nbits)
            if mode in ("burst", "hop"):
                return tol * (2 * rng.random() - 1) * np.ones(nbits)
            if mode == "bit":
                return tol * (2 * rng.random(nbits) - 1)
            raise ValueError(f"unknown tolerance mode {mode!r}")

        tolK = _draw(phase_tol, phase_tol_mode)
        btolK = _draw(bit_tol, bit_tol_mode)
        edges = np.round(np.cumsum(sps / (1.0 + btolK))).astype(int)
        lengths = np.diff(np.concatenate([[0], edges]))
        dev_eff = (1.0 + tolK) * fs / (2.0 * lengths)   # so each bit accumulates pi*(1+tol)
        f_inst = fc + np.repeat(signs * dev_eff, lengths)
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
        sig = np.cos(phase)

    # Raised-cosine amplitude ramp at both dwell edges.
    if ramp_bits and ramp_bits > 0:
        Lr = min(int(round(ramp_bits * sps)), len(sig) // 2)
        if Lr >= 2:
            rise = 0.5 - 0.5 * np.cos(np.pi * np.arange(Lr) / (Lr - 1))
            env = np.ones(len(sig))
            env[:Lr] = rise
            env[-Lr:] = rise[::-1]
            sig = sig * env
    return sig


def transmit(messages, channels=None, seed=0,
             dead_before_us=None, dead_after_us=None, ramp_bits=RAMP_BITS,
             phase_tol=0.0, phase_tol_mode="none",
             bit_tol=0.0, bit_tol_mode="none"):
    """
    Build the full real-valued RF waveform.

    messages : list of 8-char strings (sequence numbers are 1..N in order)
    channels : list of channel indices (0..5) per message; random if None
               (a random permutation, like cpfsk_tx.m 'perm' mode)
    Dead time defaults to 160..200 bit periods each side (as in cpfsk_tx.m).
    Returns  : (wave, info) where info describes the layout.
    """
    rng = np.random.default_rng(seed)
    hs = hop_set()
    n = len(messages)
    if channels is None:
        channels = rng.permutation(NCHAN)[:n].tolist()   # 'perm' mode

    # dead time in bit periods (160..200), like the MATLAB script
    if dead_before_us is None:
        dead_before = int(round(sps_dead(rng)))
    else:
        dead_before = int(round(dead_before_us * 1e-6 * FS))
    if dead_after_us is None:
        dead_after = int(round(sps_dead(rng)))
    else:
        dead_after = int(round(dead_after_us * 1e-6 * FS))

    parts = [np.zeros(dead_before)]
    dwell_starts = []
    pos = dead_before
    for i, msg in enumerate(messages):
        bits = build_frame_bits(msg, i + 1)
        sig_i = cpfsk_modulate(bits, hs[channels[i]], ramp_bits=ramp_bits,
                               phase_tol=phase_tol, phase_tol_mode=phase_tol_mode,
                               bit_tol=bit_tol, bit_tol_mode=bit_tol_mode, rng=rng)
        parts.append(sig_i)
        dwell_starts.append(pos)
        pos += len(sig_i)
    parts.append(np.zeros(dead_after))

    wave = np.concatenate(parts)
    info = {
        "channels": channels,
        "dwell_starts": dwell_starts,
        "dead_before": dead_before,
        "dead_after": dead_after,
        "hop_set": hs,
        "messages": list(messages),
    }
    return wave, info


def sps_dead(rng):
    """Dead-time length in samples: 160..200 bit periods (as in cpfsk_tx.m)."""
    return SPS * (160 + 40 * rng.random())


# ----------------------------------------------------------------------------
# Receiver stage 1: detection & channelization (STFT-based)
# ----------------------------------------------------------------------------
def detect_dwells(wave, fs=FS, threshold_frac=0.25):
    """
    Find active dwells and their channel index.

    Returns a list of dicts: {'start', 'stop', 'channel'} in samples.
    The dwells are contiguous and framed by dead time only at the two ends,
    so the active region is a single contiguous block: we locate its edges
    from a smoothed power envelope, then slice it into known-length dwells
    and identify each channel by its dominant band.
    """
    win = 4 * SPS
    env = np.convolve(wave ** 2, np.ones(win) / win, mode="same")
    peak = np.percentile(env, 99.0)
    active = env > threshold_frac * peak
    idx = np.where(active)[0]
    if len(idx) == 0:
        return []
    start = idx[0]
    stop = idx[-1] + 1

    total = stop - start
    n_dwell = int(round(total / SAMPLES_PER_FRAME))
    n_dwell = max(n_dwell, 1)

    hs = hop_set()
    dwells = []
    for k in range(n_dwell):
        a = start + k * SAMPLES_PER_FRAME
        b = a + SAMPLES_PER_FRAME
        if b > len(wave):
            break
        ch = identify_channel(wave[a:b], fs, hs)
        dwells.append({"start": a, "stop": b, "channel": ch})
    return dwells


def identify_channel(dwell, fs, hs):
    """Return the channel index whose carrier band holds the most energy."""
    N = 1 << 16
    seg = dwell[:N] if len(dwell) >= N else np.pad(dwell, (0, N - len(dwell)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / fs)
    powers = []
    for fc in hs:
        band = (freqs >= fc - HOP_SPACING / 2) & (freqs < fc + HOP_SPACING / 2)
        powers.append(spec[band].sum())
    return int(np.argmax(powers))


# ----------------------------------------------------------------------------
# Receiver stage 2: digital downconversion (DDC)
# ----------------------------------------------------------------------------
def design_ddc_lowpass(fs=FS, cutoff=12e6, numtaps=257):
    """Anti-alias / channel-select FIR for the DDC (keeps the ±4 MHz tones)."""
    return signal.firwin(numtaps, cutoff, fs=fs)


def ddc(dwell, fc, fs=FS, decim=64, fir=None):
    """
    Mix the real dwell down to complex baseband at carrier fc, low-pass, decimate.

    Returns (baseband, fs_bb, sps_bb).  decim=64 -> fs_bb=128 MHz, sps_bb=16.
    """
    if fir is None:
        fir = design_ddc_lowpass(fs)
    n = np.arange(len(dwell))
    bb = dwell * np.exp(-1j * 2 * np.pi * fc * n / fs)     # shift carrier to 0 Hz
    bb = signal.fftconvolve(bb, fir, mode="same")          # remove 2*fc image + noise
    bb = bb[::decim]                                       # decimate
    fs_bb = fs / decim
    sps_bb = SPS // decim
    return bb, fs_bb, sps_bb


# ----------------------------------------------------------------------------
# Receiver stage 3: noncoherent energy-detection demodulator
# ----------------------------------------------------------------------------
def demod_symbols(bb, fs_bb, sps_bb, offset, nbits, dev=DEV):
    """
    Noncoherent dual-tone energy detection.

    For each symbol window, compare energy at +dev vs -dev; larger wins.
    Returns (bits, confidence) where confidence = |e_up - e_dn| summed.
    """
    n = np.arange(sps_bb)
    ref_up = np.exp(-1j * 2 * np.pi * (+dev) * n / fs_bb)
    ref_dn = np.exp(-1j * 2 * np.pi * (-dev) * n / fs_bb)
    bits = []
    conf = 0.0
    for k in range(nbits):
        a = offset + k * sps_bb
        x = bb[a:a + sps_bb]
        if len(x) < sps_bb:
            x = np.pad(x, (0, sps_bb - len(x)))
        e_up = np.abs(np.sum(x * ref_up))
        e_dn = np.abs(np.sum(x * ref_dn))
        bits.append(1 if e_up > e_dn else 0)
        conf += abs(e_up - e_dn)
    return bits, conf


# ----------------------------------------------------------------------------
# Receiver stage 4: timing + frame synchronization (Barker SFD)
# ----------------------------------------------------------------------------
# The Barker-5 SFD alone (peak 5) is too short to lock reliably: a random
# payload run can match it. Instead we correlate against the full known start
# pattern PREAMBLE+SFD (13 deterministic bits), whose peak (13) random data
# cannot fake. The search is limited to the acquisition window (a few guard
# symbols + the start of the frame), never the 64-bit payload.
SYNC_WORD = PREAMBLE + SFD                 # 13-bit known frame-start pattern
SYNC_SEARCH_SYMS = 4 * 4 + len(PREAMBLE)   # guard(4) + preamble(8) + margin -> 16


def _corr_sync(bits, limit=SYNC_SEARCH_SYMS):
    """Correlate a ±1 bit stream against PREAMBLE+SFD within the acquisition
    window; return (peak, pos) where pos is the frame start (start of preamble)."""
    b = np.array([1 if x else -1 for x in bits], dtype=float)
    s = np.array([1 if x else -1 for x in SYNC_WORD], dtype=float)
    if len(b) < len(s):
        return -1e9, -1
    c = np.correlate(b, s, mode="valid")
    if limit is not None:
        c = c[:limit]
    pos = int(np.argmax(c))
    return float(c[pos]), pos


def synchronize_and_decode(bb, fs_bb, sps_bb):
    """
    Search symbol-timing offsets, demodulate, find the SFD, and extract the
    frame payload bits (HDR + PAYLOAD + CRC).

    Returns dict with keys: found, sfd_pos, timing_offset, frame_bits, corr.
    """
    best = None
    for off in range(sps_bb):
        max_syms = (len(bb) - off) // sps_bb
        bits, _ = demod_symbols(bb, fs_bb, sps_bb, off, max_syms)
        corr, pos = _corr_sync(bits)
        if best is None or corr > best["corr"]:
            best = {"corr": corr, "sfd_pos": pos, "timing_offset": off, "bits": bits}

    bits = best["bits"]
    sync_pos = best["sfd_pos"]                       # start of PREAMBLE
    hdr_start = sync_pos + len(SYNC_WORD)            # after PREAMBLE+SFD
    frame_bits = bits[hdr_start:hdr_start + HDR_BITS + PAYLOAD_BITS + CRC_BITS]
    found = len(frame_bits) == (HDR_BITS + PAYLOAD_BITS + CRC_BITS)
    return {
        "found": found,
        "sfd_pos": sync_pos + len(PREAMBLE),        # start of the SFD field
        "sync_pos": sync_pos,                       # start of the PREAMBLE
        "timing_offset": best["timing_offset"],
        "frame_bits": frame_bits,
        "corr": best["corr"],
    }


# ----------------------------------------------------------------------------
# Receiver stage 5: frame decode + integrity check
# ----------------------------------------------------------------------------
def decode_frame(frame_bits):
    """
    Decode HDR + PAYLOAD + CRC bits into (seq, message, crc_ok).
    """
    hdr = frame_bits[:HDR_BITS]
    payload = frame_bits[HDR_BITS:HDR_BITS + PAYLOAD_BITS]
    crc_rx_bits = frame_bits[HDR_BITS + PAYLOAD_BITS:]

    seq = 0
    for b in hdr:
        seq = (seq << 1) | int(b)

    chars = []
    for i in range(0, PAYLOAD_BITS, 8):
        v = 0
        for b in payload[i:i + 8]:
            v = (v << 1) | int(b)
        chars.append(chr(v))
    message = "".join(chars)

    crc_rx = 0
    for b in crc_rx_bits:
        crc_rx = (crc_rx << 1) | int(b)
    crc_calc = crc16_ccitt_false(list(hdr) + list(payload))
    crc_ok = (crc_rx == crc_calc)

    return {"seq": seq, "message": message, "crc_ok": crc_ok,
            "crc_rx": crc_rx, "crc_calc": crc_calc}


# ----------------------------------------------------------------------------
# Full receiver
# ----------------------------------------------------------------------------
def receive(wave, fs=FS, decim=64):
    """
    Complete direct-sampling receiver: detect -> DDC -> demod -> sync -> decode.
    Returns a list of per-dwell result dicts and a reassembled message list.
    """
    hs = hop_set()
    fir = design_ddc_lowpass(fs)
    dwells = detect_dwells(wave, fs)
    guard = 4 * SPS   # a few symbols of slack each side so no frame bit is lost
    results = []
    for d in dwells:
        a = max(d["start"] - guard, 0)
        b = min(d["stop"] + guard, len(wave))
        seg = wave[a:b]
        bb, fs_bb, sps_bb = ddc(seg, hs[d["channel"]], fs, decim, fir)
        sync = synchronize_and_decode(bb, fs_bb, sps_bb)
        if sync["found"]:
            dec = decode_frame(sync["frame_bits"])
        else:
            dec = {"seq": None, "message": None, "crc_ok": False}
        results.append({**d, **dec, "corr": sync["corr"]})

    valid = {r["seq"]: r["message"] for r in results if r["crc_ok"]}
    ordered = [valid[k] for k in sorted(valid.keys())]
    return results, ordered
