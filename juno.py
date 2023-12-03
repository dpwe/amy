# juno.py
# Convert juno-106 sysex patches to Amy

import amy
import javaobj
import numpy as np
import time

from dataclasses import dataclass
from typing import List

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
    pwm_lfo: bool = False
    vca_env: bool = False
    hpf: int = 0

    # These lists name the fields in the order they appear in the sysex.
    FIELDS = ['lfo_rate', 'lfo_delay_time', 'dco_lfo', 'dco_pwm', 'dco_noise', 
             'vcf_freq', 'vcf_res', 'vcf_env', 'vcf_lfo', 'vcf_kbd', 'vca_level', 
             'env_a', 'env_d', 'env_s', 'env_r', 'dco_sub']
    # After the 16 integer values, there are two bytes of bits.
    BITS1 = ['stop_16', 'stop_8', 'stop_4', 'pulse', 'triangle']
    BITS2 = ['pwm_lfo', 'vca_env', 'vcf_pos']
    
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
