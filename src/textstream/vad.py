"""Silero VAD wrapper for speech/non-speech filtering."""

from silero_vad_lite import SileroVAD

_vad = None


def _get_vad():
    global _vad
    if _vad is None:
        _vad = SileroVAD()
    return _vad


def contains_speech(audio_float32_16k, threshold: float = 0.4) -> bool:
    """Check if a 16kHz float32 mono audio chunk contains speech.

    Processes in 512-sample (32ms) windows and short-circuits on
    first detection above threshold.
    """
    vad = _get_vad()
    window = 512
    for i in range(0, len(audio_float32_16k) - window + 1, window):
        prob = vad(audio_float32_16k[i : i + window], 16000)
        if prob >= threshold:
            return True
    return False
