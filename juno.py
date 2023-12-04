# juno.py
# Convert juno-106 sysex patches to Amy

import amy
import javaobj
import numpy as np
import time

from dataclasses import dataclass
from typing import List


def to_time(midi):
  """Convert a midi value (0..127) to a time for ADSR."""
  # Range is from 10 ms to 10 sec i.e. 1e3. Return int value in ms
  #time = 0.01 * np.exp(np.log(1e3) * midi / 127.0)
  # midi 50 is ~ 1 sec
  time = 0.01 * np.exp(np.log(1e2) * midi / 50.0)
  return int(1000.0 * time)


def to_level(midi):
  # Range is from 0.001 to 1.0 i.e. 1e3
  if midi == 0:
    return 0.0
  return 0.001 * np.exp(np.log(1e3) * midi / 127.0)


def to_lfo(midi):
  # LFO frequency in Hz varies from 0.1 to 30
  return 0.1 * np.exp(np.log(300) * midi / 127.0)


def to_resonance(midi):
  # Q goes from 0.1 to 100
  return 0.1 * np.exp(np.log(1000) * midi / 127.0)


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
        to_time(self.env_a), peak_val, to_time(self.env_d),
        peak_val * to_level(self.env_s), to_time(self.env_r)
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
      vca_level = to_level(self.vca_level)
      if self.vca_gate:
        # VCA is just a gate
        vca_env_bp = "0,%f,0,0" % vca_level
      else:
        vca_env_bp = self._breakpoint_string(vca_level)
      vcf_env_polarity = -1.0 if self.vcf_neg else 1.0
      vcf_env_bp = self._breakpoint_string(to_level(vcf_env_polarity * self.vcf_env))
      osc0_args = {"osc": 0,
                   "bp0_target": amy.TARGET_AMP, "bp0": vca_env_bp,
                   "bp1_target": amy.TARGET_FILTER_FREQ, "bp1": vcf_env_bp}
      wave = amy.PULSE
      if self.triangle:
        wave = amy.SAW_UP
      osc0_args.update({"wave": wave})
      # Base VCF
      osc0_args.update({"filter_freq": to_filter_freq(self.vcf_freq),
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
