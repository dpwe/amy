// envelope.c
// VCA -- handle modulation and ADSR

#include "amy.h"

extern struct i_event* synth;
extern struct mod_event* msynth;
extern struct mod_state mglobal;


SAMPLE compute_mod_value(uint16_t mod_osc) {
    // Return the modulation-rate value for the specified oscillator.
    // i.e., this oscillator is acting as modulation for something, so
    // just calculate that modulation rate (without knowing what it
    // modulates).
    if(synth[mod_osc].wave == NOISE) return compute_mod_noise(mod_osc);
    if(synth[mod_osc].wave == SAW_DOWN) return compute_mod_saw_down(mod_osc);
    if(synth[mod_osc].wave == SAW_UP) return compute_mod_saw_up(mod_osc);
    if(synth[mod_osc].wave == PULSE) return compute_mod_pulse(mod_osc);
    if(synth[mod_osc].wave == TRIANGLE) return compute_mod_triangle(mod_osc);
    if(synth[mod_osc].wave == SINE) return compute_mod_sine(mod_osc);
    if(synth[mod_osc].wave == PCM) return compute_mod_pcm(mod_osc);
    return 0;
}

SAMPLE compute_mod_scale(uint16_t osc) {
    int16_t source = synth[osc].mod_source;
    if(synth[osc].mod_target >= 1 && source >= 0) {
        if(source != osc) {  // that would be weird
            msynth[source].amp = synth[source].amp;
            msynth[source].duty = synth[source].duty;
            msynth[source].freq = synth[source].freq;
            msynth[source].filter_freq = synth[source].filter_freq;
            msynth[source].feedback = synth[source].feedback;
            msynth[source].resonance = synth[source].resonance;
            return compute_mod_value(source);
        }
    }
    return 0; // 0 is no change, unlike bp scale
}

SAMPLE compute_breakpoint_scale(uint16_t osc, uint8_t bp_set) {

    // given a breakpoint list, compute the scale
    // we first see how many BPs are defined, and where we are in them?
    int8_t found = -1;
    int8_t release = 0;
    int32_t t1,t0;
    SAMPLE v1, v0;
    int8_t bp_r = 0;
    t0 = 0; v0 = 1.0;
    const float exponential_rate = 3.0;
    // We have to aim to overshoot to the desired gap so that we hit the target by exponential_rate time.
    const float exponential_rate_overshoot_factor = 1.0f / (1.0f - expf(-exponential_rate));
    int64_t elapsed = 0;    

    // Find out which one is release (the last one)
    
    while(synth[osc].breakpoint_times[bp_set][bp_r] >= 0 && bp_r < MAX_BREAKPOINTS) bp_r++;
    bp_r--;
    if(bp_r<0) {
        // no breakpoints, return key gate.
        SAMPLE scale = F2S(1.0f);
        if(synth[osc].note_off_clock >= 0) scale = 0;
        synth[osc].last_scale[bp_set] = scale;
        return scale; 
    }

    // Find out which BP we're in
    if(synth[osc].note_on_clock >=0) {
        elapsed = (total_samples - synth[osc].note_on_clock) + 1; 
        for(uint8_t i = 0; i < bp_r; i++) {
            if(elapsed < synth[osc].breakpoint_times[bp_set][i]) {
                // We found a segment.
                found = i;
                //i = bp_r;
                break;
            }
        }
        if(found < 0) {
            // We didn't find anything, so we are in sustain.
            found = bp_r - 1; // segment before release defines sustain
            SAMPLE scale = F2S(synth[osc].breakpoint_values[bp_set][found]);
            synth[osc].last_scale[bp_set] = scale;
            //printf("env: time %lld bpset %d seg %d SUSTAIN %f\n", total_samples, bp_set, found, S2F(scale));
            return scale;
        }
    } else if(synth[osc].note_off_clock >= 0) {
        release = 1;
        elapsed = (total_samples - synth[osc].note_off_clock) + 1; 
        // Get the last t/v pair , for release
        found = bp_r;
        int8_t bp_rx = 0;
        t0 = 0; // start the elapsed clock again
        // Release starts from wherever we got to
        v0 = synth[osc].last_scale[bp_set];
        if(elapsed > synth[osc].breakpoint_times[bp_set][bp_r]) {
            // are there multiple bp_sets? only turn off the note if it's the last one
            int32_t my_bt = synth[osc].breakpoint_times[bp_set][bp_r];
            // This is a mess but works, we could fix 
            for(uint8_t test_bp_set=0;test_bp_set<MAX_BREAKPOINT_SETS;test_bp_set++) {
                if(test_bp_set != bp_set) {
                    // Find the last bp in this bp set
                    bp_rx = 0; while(synth[osc].breakpoint_times[test_bp_set][bp_rx] >= 0 && bp_rx < MAX_BREAKPOINTS) bp_rx++; bp_rx--;
                    if(bp_rx >= 0) {
                        // If my breakpoint time is less than another breakpoint time from a different set, return 1.0 and don't end the note
                        if(my_bt < synth[osc].breakpoint_times[test_bp_set][bp_rx]) {
                            SAMPLE scale = F2S(1.0f);
                            synth[osc].last_scale[bp_set] = scale;
                            return scale;
                          }
                    }
                }
            }
            // OK. partials (et al) need a frame to fade out to avoid clicks. This is in conflict with the breakpoint release, 
            // which will set it to the bp end value before the fade out, often 0 so the fadeout never gets to hit. 
            // I'm not sure i love this solution, but PARTIAL is such a weird type that i guess having it called out like this is fine.
            if(synth[osc].wave==PARTIAL) {
                SAMPLE scale = F2S(1.0f);
                synth[osc].last_scale[bp_set] = scale;
                return scale;
            }
            synth[osc].status=OFF;
            synth[osc].note_off_clock = -1;
            SAMPLE scale = F2S(synth[osc].breakpoint_values[bp_set][bp_r]);
            synth[osc].last_scale[bp_set] = scale;
            return scale;
        }
    }

    t1 = synth[osc].breakpoint_times[bp_set][found]; 
    v1 = F2S(synth[osc].breakpoint_values[bp_set][found]);
    if(found>0 && bp_r != found && !release) {
        t0 = synth[osc].breakpoint_times[bp_set][found-1];
        v0 = F2S(synth[osc].breakpoint_values[bp_set][found-1]);
    }
    //printf("found %d bp_r %d release %d v0 %f\n", found, bp_r, release, S2F(v0));
    SAMPLE scale = v0;
    int sign = 1;
    if (v0 < 0 || v1 < 0) {
        sign = -1;
        v0 = -v0; v1 = -v1;
    }
    if(t1 < 0) { // || v1 < 0) {
        scale = 0;
    } else if(t1==t0 || elapsed==t1) {
        // This way we return exact zero for v1 at the end of the segment, rather than BREAKPOINT_EPS
        scale = v1;
    } else {
        // OK, we are transition from v0 to v1 , and we're at elapsed time between t0 and t1
        float time_ratio = ((float)(elapsed - t0) / (float)(t1 - t0));
        // Compute scale based on which type we have
        if(synth[osc].breakpoint_target[bp_set] & TARGET_LINEAR) {
            scale = v0 + MUL4_SS(v1 - v0, F2S(time_ratio));
        } else if(synth[osc].breakpoint_target[bp_set] & TARGET_TRUE_EXPONENTIAL) {
            v0 = MAX(v0, F2S(BREAKPOINT_EPS));
            v1 = MAX(v1, F2S(BREAKPOINT_EPS));
            float dx7_exponential_rate = -logf(S2F(v1)/S2F(v0)) / (t1 - t0);
            scale = MUL4_SS(v0, F2S(expf(-dx7_exponential_rate * (elapsed - t0))));
        } else if(synth[osc].breakpoint_target[bp_set] & TARGET_DX7_EXPONENTIAL) {
            // Somewhat complicated relationship, see https://colab.research.google.com/drive/1qZmOw4r24IDijUFlel_eSoWEf3L5VSok#scrollTo=F5zkeACrOlum
#define LINEAR_TO_DX7_LEVEL(linear) (MIN(99.0, logf(MAX(BREAKPOINT_EPS, linear)) * 8.0 + 99.0))
#define DX7_LEVEL_TO_LINEAR(level) (powf(2.0, (level - 99.0)/8.0))
#define MIN_LEVEL 34
#define ATTACK_RANGE 75
#define MAP_ATTACK_LEVEL(level) (1 - MAX(level - MIN_LEVEL, 0) / ATTACK_RANGE)
            float mapped_current_level = MAP_ATTACK_LEVEL(LINEAR_TO_DX7_LEVEL(S2F(v0)));
            float mapped_target_level = MAP_ATTACK_LEVEL(LINEAR_TO_DX7_LEVEL(S2F(v1)));
            float t_const = (t1 - t0) / logf(mapped_current_level / mapped_target_level);
            float my_t0 = -t_const * logf(mapped_current_level);
            if (v1 > v0) {
                // This is the magic equation that shapes the DX7 attack envelopes.
                scale = F2S(DX7_LEVEL_TO_LINEAR(MIN_LEVEL + ATTACK_RANGE * (1 - expf(-(my_t0 + elapsed)/t_const))));
            } else {
                // Decay is regular true_exponential
                v0 = MAX(v0, F2S(BREAKPOINT_EPS));
                v1 = MAX(v1, F2S(BREAKPOINT_EPS));
                float dx7_exponential_rate = -logf(S2F(v1)/S2F(v0)) / (t1 - t0);
                scale = MUL4_SS(v0, F2S(expf(-dx7_exponential_rate * (elapsed - t0))));
            }
        } else { // "false exponential?"
            // After the full amount of time, the exponential decay will reach (1 - expf(-3)) = 0.95
            // so make the target gap a little bit bigger, to ensure we meet v1
            scale = v0 + MUL4_SS(v1 - v0, F2S(exponential_rate_overshoot_factor * (1.0f - expf(-exponential_rate * time_ratio))));
            //printf("false_exponential time %lld bpset %d seg %d time_ratio %f scale %f\n", total_samples, bp_set, found, time_ratio, S2F(scale));
        }
    }
    // If sign is negative, render the env as 1 - ADSR.
    if (sign < 0) {
        scale = F2S(1.0f) - scale;
    }
    // Keep track of the most-recently returned non-release scale.
    if (!release) synth[osc].last_scale[bp_set] = scale;
    //printf("env: time %lld bpset %d seg %d t0 %d t1 %d elapsed %lld v0 %f v1 %f scale %f\n", total_samples, bp_set, found, t0, t1, elapsed, S2F(v0), S2F(v1), S2F(scale));
    return scale;
}

