# juno.py
# Convert juno-106 sysex patches to Amy

import amy
import javaobj
import numpy as np
import time

from dataclasses import dataclass
from typing import List


  # Range is from 10 ms to 12 sec i.e. 1200.
  # (12 sec is allegedly the max decay time of the EG, see
  # page 32 of Juno-106 owner's manual,
  # https://cdn.roland.com/assets/media/pdf/JUNO-106_OM.pdf .)
  # Return int value in ms
  #time = 0.01 * np.exp(np.log(1e3) * midi / 127.0)
  # midi 30 is ~ 200 ms, 50 is ~ 1 sec, so
  #        D=30 
  # 
  # from demo at https://www.synthmania.com/Roland%20Juno-106/Audio/Juno-106%20Factory%20Preset%20Group%20A/14%20Flutes.mp3
  # A11 Brass set        A=3  D=49  S=45 R=32 -> 
  # A14 Flute            A=23 D=81  S=0  R=18 -> A=200ms, R=200ms, D=0.22 to 0.11 in 0.830 s / 0.28 to 0.14 in 0.92     R = 0.2 to 0.1 in 0.046
  # A16 Brass & Strings  A=44 D=66  S=53 R=44 -> A=355ms, R=        
  # A15 Moving Strings   A=13 D=87       R=35 -> A=100ms, R=600ms,
  # A26 Celeste          A=0  D=44  S=0  R=81 -> A=2ms             D=0.48 to 0.24 in 0.340 s                           R = 0.9 to 0.5 in 0.1s; R clearly faster than D
  # A27 Elect Piano      A=1  D=85  S=43 R=40 -> A=14ms,  R=300ms
  # A28 Elect. Piano II  A=0  D=68  S=0  R=22 ->                   D=0.30 to 0.15 in 0.590 s  R same as D?
  # A32 Steel Drums      A=0  D=26  S=0  R=37 ->                   D=0.54 to 0.27 in 0.073 s
  # A34 Brass III        A=58 D=100 S=94 R=37 -> A=440ms, R=1000ms
  # A35 Fanfare          A=72 D=104 S=75 R=49 -> A=600ms, R=1200ms 
  # A37 Pizzicato        A=0  D=11  S=0  R=12 -> A=6ms,   R=86ms   D=0.66 to 0.33 in 0.013 s
  # A41 Bass Clarinet    A=11 D=75  S=0  R=25 -> A=92ms,  R=340ms, D=0.20 to 0.10 in 0.820 s /                            R = 0.9 to 0.45 in 0.070
  # A42 English Horn     A=8  D=81  S=21 R=16 -> A=68ms,  R=240ms,
  # A45 Koto             A=0  D=56  S=0  R=39 ->                   D=0.20 to 0.10 in 0.160 s
  # A46 Dark Pluck       A=0  D=52  S=15 R=63 ->
  # A48 Synth Bass I     A=0  D=34  S=0  R=36 ->                   D=0.60 to 0.30 in 0.096 s
  # A56 Funky III        A=0  D=24  S=0  R=2                       D 1/2 in 0.206
  # A61 Piano II         A=0  D=98  S=0  R=32                      D 1/2 in 1.200
  # A  0   1   8   11   13   23   44  58
  # ms 6   14  68  92   100  200  355 440
  # D            11  24  26  34  44  56  68  75  81  98
  # 1/2 time ms  13  206 73  96  340 160 590 830 920 1200
  # R  12  16  18  25   35   37   40
  # ms 86  240 200 340  600  1000 300
  
def to_decay_time(midi):
  """Convert a midi value (0..127) to a time for ADSR."""
  time = 12 * np.exp(np.log(120) * midi/100)
  # time is time to decay to 1/2; Amy envelope times are to decay to exp(-3) = 0.05
  # 
  return np.log(0.05) / np.log(0.5) * time


def to_release_time(midi):
  """Convert a midi value (0..127) to a time for ADSR."""
  time = 100 * np.exp(np.log(16) * midi/100)
  return np.log(0.05) / np.log(0.5) * time


def to_attack_time(midi):
  """Convert a midi value (0..127) to a time for ADSR."""
  return 6 + 8 * midi


def to_level(midi):
  # Map midi to 0..1, linearly.
  return midi / 127.0


def level_to_amp(level):
  # level is 0.0 to 1.0; amp is 0.001 to 1.0
  if level == 0.0:
    return 0.0
  return 0.001 * np.exp(level * np.log(1000.0))

def to_lfo(midi):
  # LFO frequency in Hz varies from 0.1 to 30
  return 0.1 * np.exp(np.log(300) * midi / 127.0)


def to_resonance(midi):
  # Q goes from 0.1 to 30
  return 0.1 * np.exp(np.log(300) * midi / 127.0)


def to_filter_freq(midi):
  # filter_freq goes from ? 100 to 6400 Hz with 18 steps/octave
  return 100 * np.exp(np.log(2) * midi / 20.0)


@dataclass
class JunoPatch:
    """Encapsulates information in a Juno Patch."""
    name: str = ""
    lfo_rate: int = 0
    lfo_delay_time: int = 0
    dco_lfo: int = 0
    dco_pwm: int = 0
    dco_noise: int = 0
    vcf_freq: int = 0
    vcf_res: int = 0
    vcf_env: int = 0
    vcf_lfo: int = 0
    vcf_kbd: int = 0
    vca_level: int = 0
    env_a: int = 0
    env_d: int = 0
    env_s: int = 0
    env_r: int = 0
    dco_sub: int = 0
    stop_16: bool = False
    stop_8: bool = False
    stop_4: bool = False
    pulse: bool = False
    triangle: bool = False
    chorus: int = 0
    pwm_manual: bool = False  # else lfo
    vca_gate: bool = False  # else env
    vcf_neg: bool = False  # else pos
    hpf: int = 0

    # These lists name the fields in the order they appear in the sysex.
    FIELDS = ['lfo_rate', 'lfo_delay_time', 'dco_lfo', 'dco_pwm', 'dco_noise', 
             'vcf_freq', 'vcf_res', 'vcf_env', 'vcf_lfo', 'vcf_kbd', 'vca_level', 
             'env_a', 'env_d', 'env_s', 'env_r', 'dco_sub']
    # After the 16 integer values, there are two bytes of bits.
    BITS1 = ['stop_16', 'stop_8', 'stop_4', 'pulse', 'triangle']
    BITS2 = ['pwm_manual', 'vcf_neg', 'vca_gate']
    
    @staticmethod
    def from_patch_number(patch_number):
      pobj = javaobj.load(open('juno106_factory_patches.ser', 'rb'))
      patch = pobj.v.elementData[patch_number]
      return JunoPatch.from_sysex(bytes(patch.sysex), name=patch.name)

    @classmethod
    def from_sysex(cls, sysexbytes, name=None):
        """Decode sysex bytestream into JunoPatch fields."""
        assert len(sysexbytes) == 18
        result = JunoPatch(name=name)
        # The first 16 bytes are sliders.
        for index, field in enumerate(cls.FIELDS):
            setattr(result, field, int(sysexbytes[index]))
        # Then there are two bytes of switches.
        for index, field in enumerate(cls.BITS1):
            setattr(result, field, (int(sysexbytes[16]) & (1 << index)) > 0)
        setattr(result, 'chorus', int(sysexbytes[16]) >> 5)
        for index, field in enumerate(cls.BITS2):
            setattr(result, field, (int(sysexbytes[17]) & (1 << index)) > 0)
        setattr(result, 'hpf', int(sysexbytes[17]) >> 3)
        return result

    def _breakpoint_string(self, peak_val):
      """Format a breakpoint string from the ADSR parameters reaching a peak."""
      return "0,0,%d,%f,%d,%f,%d,0" % (
        to_attack_time(self.env_a), peak_val, to_decay_time(self.env_d),
        peak_val * to_level(self.env_s), to_release_time(self.env_r)
      )
  
    def send_to_AMY(self):
      """Output AMY commands to set up the patch.
      Send amy.send(vel=0,osc=6,note=50) afterwards."""
      amy.reset()
      # osc 0 is main osc (pwm/triangle)
      #   env0 is VCA
      #   env1 is VCF
      # osc 1 is LFO
      #   LFO can hit PWM or pitch (or VCF) but we can't scale separately for each.
      # osc 2 is the sub-oscillator (square) or noise?  Do we need both?
      vca_level = level_to_amp(to_level(self.vca_level))
      if self.vca_gate:
        # VCA is just a gate
        vca_env_bp = "0,%f,0,0" % vca_level
      else:
        vca_env_bp = self._breakpoint_string(vca_level)
      vcf_env_polarity = -1.0 if self.vcf_neg else 1.0
      # Juno interprets VCF env is as over and above filter_freq, where 1.0 = many octaves above
      # AMY *now* interprents VCF env as filter_freq * (1 + env)
      # Set filter_freq so that vcf env peak is 64x for 1.0
      filter_freq = to_filter_freq(self.vcf_freq)
      vcf_env_bp = self._breakpoint_string(vcf_env_polarity * (2.0 ** (6.0 * to_level(self.vcf_env))))
      osc0_args = {"osc": 0,
                   "bp0_target": amy.TARGET_AMP, "bp0": vca_env_bp,
                   "bp1_target": amy.TARGET_FILTER_FREQ, "bp1": vcf_env_bp}
      wave = amy.PULSE
      if self.triangle:
        wave = amy.SAW_UP
      osc0_args.update({"wave": wave})
      # Base VCF
      osc0_args.update({"filter_freq": filter_freq,
                        "filter_type": amy.FILTER_LPF,
                        "resonance": to_resonance(self.vcf_res)})
      
      lfo_args = {}
      pwm_lfo = 0 if self.pwm_manual else self.dco_pwm
      lfo_amp = np.max(np.array([self.dco_lfo, self.vcf_lfo, pwm_lfo]))
      if lfo_amp:
        lfo_target = 0
        if self.dco_lfo > lfo_amp / 2:
          lfo_target |= amy.TARGET_FREQ
        if self.vcf_lfo > lfo_amp / 2:
          lfo_target |= amy.TARGET_FILTER_FREQ
        if pwm_lfo > lfo_amp / 2:
          lfo_target |= amy.TARGET_DUTY
        lfo_env_bp = "0,0,%d,%f,%d,0" % (
          to_time(self.lfo_delay_time), to_level(lfo_target), to_time(self.env_r)
        )
        osc0_args.update({"mod_source": 1, "mod_target": lfo_target})
        lfo_args = {'osc': 1, 'bp0_target': amy.TARGET_AMP, 'bp0': lfo_env_bp,
                    'wave': TRIANGLE, 'freq': to_lfo(self.lfo_rate)}
                                           
      # Sub-harmonic and noise oscillators have the same VCA, VCF, and LFO params as main osc.
      # ...

      # Send it all out.
      print(osc0_args, lfo_args)
      amy.send(**osc0_args)
      if lfo_args:
        amy.send(**lfo_args)

# To do:
#  - make the filter env be filter_freq * (1 + filter ADSR)
#    - undo the (1 - env) output for negative envs
#  - filter ADSR scaling needs to be stretched.  Brass sysex has vcf_freq = 35 (-> 356 Hz) vcf_env = 58 (-> 0.023), actual frequency should be ~3000 to 600
#    so vcf_env 58 should be 2-3 octaves
#    and vcf_freq 35 should be ~600 Hz (midi 35 is 62 Hz, so we need a narrower range, e.g. 24 steps/oct, so entire 127 range is 5 octaves = 32x = 100 to 3200 Hz - too small
#    vcf needs to cover 50 to 8000 Hz, so 160x or 7+ octaves, 18 steps/octave
#  VCF ADSR wants to be added to VCF base_freq before exponentiation.. convert freq repn to logf internally
