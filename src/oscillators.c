#include "amy.h"


// For checking assumptions about bitwidths.
#include <assert.h>

#include "sine_lutset_fxpt.h"
#include "impulse_lutset_fxpt.h"
#include "triangle_lutset_fxpt.h"

// For hardware random on ESP
#ifdef ESP_PLATFORM
#include <esp_system.h>
#endif





/* Dan Ellis libblosca functions */
const LUT *choose_from_lutset(float period, const LUT *lutset) {
    // Select the best entry from a lutset for a given period. 
    //
    // Args:
    //    period: (float) Target period of waveform in fractional samples.
    //    lutset: Sorted list of LUTs, as generated by create_lutset().
    //
    // Returns:
    //   One of the LUTs from the lutset, best suited to interpolating to generate
    //   a waveform of the desired period.
    // Use the earliest (i.e., longest, most harmonics) LUT that works
    // (i.e., will not actually cause aliasing).
    // So start with the highest-bandwidth (and longest) LUTs, but skip them
    // if they result in aliasing.
    const LUT *lut_table = NULL;
    int lut_size = 0;
    int lut_index = 0;
    while(lutset[lut_index].table_size > 0) {
        lut_table = &lutset[lut_index];
        lut_size = lutset[lut_index].table_size;
        // What proportion of nyquist does the highest harmonic in this table occupy?
        float lut_bandwidth = 2 * lutset[lut_index].highest_harmonic / (float)lut_size;
        // To complete one cycle of <lut_size> points in <period> steps, each step
        // will need to be this many samples:
        float lut_hop = lut_size / period;
        // If we have a signal with a given bandwidth, but then speed it up by 
        // skipping lut_hop samples per sample, its bandwidth will increase 
        // proportionately.
        float interp_bandwidth = lut_bandwidth * lut_hop;
        if (interp_bandwidth < 0.9) {
            // No aliasing, even with a 10% buffer (i.e., 19.8 kHz).
            break;
        }
        ++lut_index;
    }
    // At this point, we either got to the end of the LUT table, or we found a
    // table we could interpolate without aliasing.

    return lut_table;
}

PHASOR render_lut_fm_osc(SAMPLE* buf,
                         PHASOR phase, 
                         PHASOR step,
                         SAMPLE incoming_amp, SAMPLE ending_amp,
                         const LUT* lut,
                         SAMPLE* mod, SAMPLE feedback_level, SAMPLE* last_two) { 
    //printf("render_fm: phase %f step %f a0 %f a1 %f lut_sz %d mod 0x%lx fb %f l2 0x%lx\n",
    //       P2F(phase), P2F(step), S2F(incoming_amp), S2F(ending_amp), lut->table_size,
    //       mod, S2F(feedback_level), last_two);
    //if(mod)
    //    printf("render_fm: i_amp %f mod[:5]= %f %f %f %f %f\n", S2F(incoming_amp),
    //           S2F(mod[0]), S2F(mod[1]), S2F(mod[2]), S2F(mod[3]), S2F(mod[4]));
    int lut_mask = lut->table_size - 1;
    int lut_bits = lut->log_2_table_size;
    SAMPLE past0 = 0, past1 = 0, sample = 0;
    if(last_two) {  // Only for FM oscillators.
        // Setup so that first pas through feedback_level block moves these into past0 and past1.
        sample = last_two[0];
        past0 = last_two[1];
    }
    SAMPLE current_amp = incoming_amp;
    SAMPLE incremental_amp = (ending_amp - incoming_amp) >> BLOCK_SIZE_BITS; // i.e. delta(amp) / BLOCK_SIZE
    for(uint16_t i = 0; i < AMY_BLOCK_SIZE; i++) {
        // total_phase can extend beyond [0, 1) but we apply lut_mask before we use it.
        PHASOR total_phase = phase;

        if(mod) total_phase += S2P(mod[i]);
        if(feedback_level) {
            past1 = past0;
            past0 = sample;   // Feedback is taken before output scaling.
            total_phase += S2P(MUL4_SS(feedback_level, (past1 + past0) >> 1));
        }
        int16_t base_index = INT_OF_P(total_phase, lut_bits);
        SAMPLE frac = S_FRAC_OF_P(total_phase, lut_bits);
        LUTSAMPLE b = lut->table[base_index];
        LUTSAMPLE c = lut->table[(base_index + 1) & lut_mask];
        SAMPLE sample = L2S(b) + MUL0_SS(L2S(c - b), frac);
        buf[i] += MUL4_SS(current_amp, sample);
        current_amp += incremental_amp;
        phase = P_WRAPPED_SUM(phase, step);
    }
    if(last_two) {
        last_two[0] = sample;
        last_two[1] = past0;
    }
    return phase;
}

PHASOR render_lut(SAMPLE* buf,
                  PHASOR phase, PHASOR step,
                  SAMPLE incoming_amp, SAMPLE ending_amp,
                  const LUT* lut) {
    return render_lut_fm_osc(buf, phase, step, incoming_amp, ending_amp, lut,
                             NULL, 0, NULL);
}

void lpf_buf(SAMPLE *buf, SAMPLE decay, SAMPLE *state) {
    // Implement first-order low-pass (leaky integrator).
    SAMPLE s = *state;
    for (uint16_t i = 0; i < AMY_BLOCK_SIZE; ++i) {
        buf[i] += MUL4_SS(decay, s);
        s = buf[i];
    }
    *state = s;
}


/* Pulse wave */
void pulse_note_on(uint16_t osc) {
    //printf("pulse_note_on: time %lld osc %d freq %f amp %f last_amp %f\n", total_samples, osc, synth[osc].freq, S2F(synth[osc].amp), S2F(synth[osc].last_amp));
    float period_samples = (float)AMY_SAMPLE_RATE / synth[osc].freq;
    synth[osc].lut = choose_from_lutset(period_samples, impulse_fxpt_lutset);
    // Tune the initial integrator state to compensate for mid-sample alignment of table.
    float float_amp = S2F(synth[osc].amp) * synth[osc].freq * 4.0f / AMY_SAMPLE_RATE;
    synth[osc].lpf_state = MUL4_SS(F2S(-0.5 * float_amp), L2S(synth[osc].lut->table[0]));
}

void render_lpf_lut(SAMPLE* buf, uint16_t osc, float duty, int8_t direction, SAMPLE dc_offset) {
    // Common function for pulse and saw.
    PHASOR step = F2P(msynth[osc].freq / (float)AMY_SAMPLE_RATE);  // cycles per sec / samples per sec -> cycles per sample
    // LPF time constant should be ~ 10x osc period, so droop is minimal.
    // alpha = 1 - 1 / t_const; t_const = 10 / m_freq, so alpha = 1 - m_freq / 10
    synth[osc].lpf_alpha = F2S(1.0f - msynth[osc].freq / (10.0f * AMY_SAMPLE_RATE));
    // Scale the impulse proportional to the phase increment step so its integral remains ~constant.
    const LUT *lut = synth[osc].lut;
    SAMPLE amp = direction * MUL4_SS(msynth[osc].amp, F2S(P2F(step) * 4.0f * lut->scale_factor));
    synth[osc].phase = render_lut(buf, synth[osc].phase, step, synth[osc].last_amp, amp, lut);
    if (duty > 0) {  // For pulse only, add a second delayed negative LUT wave.
        PHASOR pwm_phase = P_WRAPPED_SUM(synth[osc].phase, F2P(duty));
        render_lut(buf, pwm_phase, step, -synth[osc].last_amp, -amp, synth[osc].lut);
    }
    if (dc_offset) {
        // For saw only, apply a dc shift so integral is ~0.
        // But we have to apply the linear amplitude env on top as well, copying the way it's done in render_lut.
        SAMPLE current_amp = synth[osc].last_amp;
        SAMPLE incremental_amp = (amp - synth[osc].last_amp) >> BLOCK_SIZE_BITS; // i.e. delta(amp) / BLOCK_SIZE
        for (int i = 0; i < AMY_BLOCK_SIZE; ++i) {
            buf[i] += MUL4_SS(current_amp, dc_offset);
            current_amp += incremental_amp;
        }
    }        
    // LPF to integrate to convert pair of (+, -) impulses into a rectangular wave.
    lpf_buf(buf, synth[osc].lpf_alpha, &synth[osc].lpf_state);
    // Remember last_amp.
    synth[osc].last_amp = amp;
}

void render_pulse(SAMPLE* buf, uint16_t osc) {
    // Second (negative) impulse is <duty> cycles later.
    float duty = msynth[osc].duty;
    if (duty < 0.01f) duty = 0.01f;
    if (duty > 0.99f) duty = 0.99f;

    render_lpf_lut(buf, osc, duty, 1, 0);
}

void pulse_mod_trigger(uint16_t osc) {
    //float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;
    //float period = 1. / (synth[osc].freq/mod_sr);
    //synth[osc].step = period * synth[osc].phase;
}

// dpwe sez to use this method for low-freq mod pulse still 
SAMPLE compute_mod_pulse(uint16_t osc) {
    // do BW pulse gen at SR=44100/64
    if(msynth[osc].duty < 0.001f || msynth[osc].duty > 0.999) msynth[osc].duty = 0.5;
    if(synth[osc].phase >= F2P(msynth[osc].duty)) {
        synth[osc].sample = F2S(1.0f);
    } else {
        synth[osc].sample = F2S(-1.0f);
    }
    float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;  // samples per sec / samples per call = calls per sec
    synth[osc].phase = P_WRAPPED_SUM(synth[osc].phase, F2P(msynth[osc].freq / mod_sr));  // cycles per sec / calls per sec = cycles per call
    return MUL4_SS(synth[osc].sample, msynth[osc].amp);
}


/* Saw waves */
void saw_note_on(uint16_t osc, int8_t direction_notused) {
    //printf("saw_note_on: time %lld osc %d freq %f amp %f last_amp %f phase %f\n", total_samples, osc, synth[osc].freq, S2F(synth[osc].amp), S2F(synth[osc].last_amp), P2F(synth[osc].phase));
    float period_samples = ((float)AMY_SAMPLE_RATE / synth[osc].freq);
    synth[osc].lut = choose_from_lutset(period_samples, impulse_fxpt_lutset);
    // Calculate the mean of the LUT.
    SAMPLE lut_sum = 0;
    for (int i = 0; i < synth[osc].lut->table_size; ++i) {
        lut_sum += L2S(synth[osc].lut->table[i]);
    }
    int lut_bits = synth[osc].lut->log_2_table_size;
    synth[osc].dc_offset = -(lut_sum >> lut_bits);
    synth[osc].lpf_state = 0;
    synth[osc].last_amp = 0;
}

void saw_down_note_on(uint16_t osc) {
    saw_note_on(osc, -1);
}
void saw_up_note_on(uint16_t osc) {
    saw_note_on(osc, 1);
}

void render_saw(SAMPLE* buf, uint16_t osc, int8_t direction) {
    render_lpf_lut(buf, osc, 0, direction, synth[osc].dc_offset);
    //printf("render_saw: time %lld osc %d buf[]=%f %f %f %f %f %f %f %f\n",
    //       total_samples, osc, S2F(buf[0]), S2F(buf[1]), S2F(buf[2]), S2F(buf[3]), S2F(buf[4]), S2F(buf[5]), S2F(buf[6]), S2F(buf[7]));
}

void render_saw_down(SAMPLE* buf, uint16_t osc) {
    render_saw(buf, osc, -1);
}
void render_saw_up(SAMPLE* buf, uint16_t osc) {
    render_saw(buf, osc, 1);
}


void saw_mod_trigger(uint16_t osc) {
    //float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;
    //float period = 1. / (synth[osc].freq/mod_sr);
    //synth[osc].step = period * synth[osc].phase;
}

void saw_up_mod_trigger(uint16_t osc) {
    saw_mod_trigger(osc);
}
void saw_down_mod_trigger(uint16_t osc) {
    saw_mod_trigger(osc);
}

// TODO -- this should use dpwe code
SAMPLE compute_mod_saw(uint16_t osc, int8_t direction) {
    // Saw waveform is just the phasor.
    synth[osc].sample = (P2S(synth[osc].phase) << 1) - F2S(1.0f);
    float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;  // samples per sec / samples per call = calls per sec
    synth[osc].phase = P_WRAPPED_SUM(synth[osc].phase, F2P(msynth[osc].freq / mod_sr));  // cycles per sec / calls per sec = cycles per call
    return MUL4_SS(synth[osc].sample, direction * msynth[osc].amp);
}

SAMPLE compute_mod_saw_down(uint16_t osc) {
    return compute_mod_saw(osc, -1);
}

SAMPLE compute_mod_saw_up(uint16_t osc) {
    return compute_mod_saw(osc, 1);
}



/* triangle wave */
void triangle_note_on(uint16_t osc) {
    float period_samples = (float)AMY_SAMPLE_RATE / synth[osc].freq;
    synth[osc].lut = choose_from_lutset(period_samples, triangle_fxpt_lutset);
}

void render_triangle(SAMPLE* buf, uint16_t osc) {
    PHASOR step = F2P(msynth[osc].freq / (float)AMY_SAMPLE_RATE);  // cycles per sec / samples per sec -> cycles per sample
    SAMPLE amp = msynth[osc].amp;
    synth[osc].phase = render_lut(buf, synth[osc].phase, step, synth[osc].last_amp, amp, synth[osc].lut);
    synth[osc].last_amp = amp;
}

void triangle_mod_trigger(uint16_t osc) {
    // float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;
    // float period = 1. / (synth[osc].freq/mod_sr);
    // synth[osc].step = period * synth[osc].phase;
}

// TODO -- this should use dpwe code 
SAMPLE compute_mod_triangle(uint16_t osc) {
    // Saw waveform is just the phasor.
    SAMPLE sample = P2S(synth[osc].phase) << 2;  // 0..4
    if (sample > F2S(2.0f))  sample = F2S(4.0f) - sample;  // 0..2..0
    synth[osc].sample = sample - F2S(1.0f);  // -1 .. 1
    float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;  // samples per sec / samples per call = calls per sec
    synth[osc].phase = P_WRAPPED_SUM(synth[osc].phase, F2P(msynth[osc].freq / mod_sr));  // cycles per sec / calls per sec = cycles per call
    return MUL4_SS(synth[osc].sample, msynth[osc].amp);
}

extern int64_t total_samples;

/* FM */
// NB this uses new lingo for step, skip, phase etc
void fm_sine_note_on(uint16_t osc, uint16_t algo_osc) {
    if(synth[osc].ratio >= 0) {
        msynth[osc].freq = (msynth[algo_osc].freq * synth[osc].ratio);
    }
    // An empty exercise since there is only one entry in sine_lutset.
    float period_samples = (float)AMY_SAMPLE_RATE / msynth[osc].freq;
    synth[osc].lut = choose_from_lutset(period_samples, sine_fxpt_lutset);
}

void render_fm_sine(SAMPLE* buf, uint16_t osc, SAMPLE* mod, SAMPLE feedback_level, uint16_t algo_osc, SAMPLE mod_amp) {
    if(synth[osc].ratio >= 0) {
        msynth[osc].freq = msynth[algo_osc].freq * synth[osc].ratio;
    }
    PHASOR step = F2P(msynth[osc].freq / (float)AMY_SAMPLE_RATE);  // cycles per sec / samples per sec -> cycles per sample
    SAMPLE amp = MUL4_SS(msynth[osc].amp, mod_amp);
    synth[osc].phase = render_lut_fm_osc(buf, synth[osc].phase, step,
                                         synth[osc].last_amp, amp, 
                                         synth[osc].lut,
                                         mod, feedback_level, synth[osc].last_two);
    synth[osc].last_amp = amp;
}

/* sine */
void sine_note_on(uint16_t osc) {
    //printf("sine_note_on: osc %d freq %f\n", osc, synth[osc].freq);
    // There's really only one sine table, but for symmetry with the other ones...
    float period_samples = (float)AMY_SAMPLE_RATE / synth[osc].freq;
    synth[osc].lut = choose_from_lutset(period_samples, sine_fxpt_lutset);
}

void render_sine(SAMPLE* buf, uint16_t osc) { 
    PHASOR step = F2P(msynth[osc].freq / (float)AMY_SAMPLE_RATE);  // cycles per sec / samples per sec -> cycles per sample
    SAMPLE amp = msynth[osc].amp;
    //printf("render_sine: osc %d freq %f amp %f\n", osc, P2F(step), S2F(amp));
    synth[osc].phase = render_lut(buf, synth[osc].phase, step, synth[osc].last_amp, amp, synth[osc].lut);
    synth[osc].last_amp = amp;
}


// TOOD -- not needed anymore
SAMPLE compute_mod_sine(uint16_t osc) { 
    // One sample pulled out of render_lut.
    const LUT *lut = synth[osc].lut;
    int lut_mask = lut->table_size - 1;
    int lut_bits = lut->log_2_table_size;
    int16_t base_index = INT_OF_P(synth[osc].phase, lut_bits);
    SAMPLE frac = S_FRAC_OF_P(synth[osc].phase, lut_bits);
    LUTSAMPLE b = lut->table[base_index];
    LUTSAMPLE c = lut->table[(base_index + 1) & lut_mask];
    synth[osc].sample = L2S(b) + MUL0_SS(L2S(c - b), frac);
    float mod_sr = (float)AMY_SAMPLE_RATE / (float)AMY_BLOCK_SIZE;  // samples per sec / samples per call = calls per sec
    synth[osc].phase = P_WRAPPED_SUM(synth[osc].phase, F2P(msynth[osc].freq / mod_sr));  // cycles per sec / calls per sec = cycles per call
    return MUL4_SS(synth[osc].sample, msynth[osc].amp);
}

void sine_mod_trigger(uint16_t osc) {
    sine_note_on(osc);
}

// Returns a SAMPLE between -1 and 1.
SAMPLE amy_get_random() {
    assert(RAND_MAX == 2147483647); // 2^31 - 1
    return rand() >> (31 - S_FRAC_BITS);
}

/* noise */

void render_noise(SAMPLE *buf, uint16_t osc) {
    for(uint16_t i=0;i<AMY_BLOCK_SIZE;i++) {
        buf[i] = MUL4_SS(amy_get_random(), msynth[osc].amp);
    }
}

SAMPLE compute_mod_noise(uint16_t osc) {
    return MUL4_SS(amy_get_random(), msynth[osc].amp);
}



/* partial */

#if AMY_HAS_PARTIALS == 1

void render_partial(SAMPLE * buf, uint16_t osc) {
    PHASOR step = F2P(msynth[osc].freq / (float)AMY_SAMPLE_RATE);  // cycles per sec / samples per sec -> cycles per sample
    SAMPLE amp = msynth[osc].amp;
    synth[osc].phase = render_lut(buf, synth[osc].phase, step, synth[osc].last_amp, amp, synth[osc].lut);
    synth[osc].last_amp = amp;
}

void partial_note_on(uint16_t osc) {
    float period_samples = (float)AMY_SAMPLE_RATE / msynth[osc].freq;
    synth[osc].lut = choose_from_lutset(period_samples, sine_fxpt_lutset);
}

void partial_note_off(uint16_t osc) {
    synth[osc].substep = 2;
    synth[osc].note_on_clock = -1;
    synth[osc].note_off_clock = total_samples;   
    synth[osc].last_amp = 0;
    synth[osc].status=OFF;
}

#endif

#if AMY_KS_OSCS > 0

#define MAX_KS_BUFFER_LEN 802 // 44100/55  -- 55Hz (A1) lowest we can go for KS
SAMPLE ** ks_buffer; 
uint8_t ks_polyphony_index; 


/* karplus-strong */

void render_ks(SAMPLE * buf, uint16_t osc) {
    SAMPLE half = MUL0_SS(F2S(0.5f),synth[osc].feedback); 
    if(msynth[osc].freq >= 55) { // lowest note we can play
        uint16_t buflen = (uint16_t)(AMY_SAMPLE_RATE / msynth[osc].freq);
        for(uint16_t i=0;i<AMY_BLOCK_SIZE;i++) {
            uint16_t index = (uint16_t)(synth[osc].step);
            synth[osc].sample = ks_buffer[ks_polyphony_index][index];
            ks_buffer[ks_polyphony_index][index] =                 
                MUL4_SS(
                    (ks_buffer[ks_polyphony_index][index] + ks_buffer[ks_polyphony_index][(index + 1) % buflen]),
                    half);
            synth[osc].step = (index + 1) % buflen;
            buf[i] = MUL4_SS(synth[osc].sample, msynth[osc].amp);
        }
    }
}

void ks_note_on(uint16_t osc) {
    if(msynth[osc].freq<=0) msynth[osc].freq = 1;
    uint16_t buflen = (uint16_t)(AMY_SAMPLE_RATE / msynth[osc].freq);
    if(buflen > MAX_KS_BUFFER_LEN) buflen = MAX_KS_BUFFER_LEN;
    // init KS buffer with noise up to max
    for(uint16_t i=0;i<buflen;i++) {
        ks_buffer[ks_polyphony_index][i] = amy_get_random();
    }
    ks_polyphony_index++;
    if(ks_polyphony_index == AMY_KS_OSCS) ks_polyphony_index = 0;
}

void ks_note_off(uint16_t osc) {
    msynth[osc].amp = 0;
}


void ks_init(void) {
    // 6ms buffer
    ks_polyphony_index = 0;
    ks_buffer = (SAMPLE**) malloc(sizeof(SAMPLE*)*AMY_KS_OSCS);
    for(int i=0;i<AMY_KS_OSCS;i++) ks_buffer[i] = (SAMPLE*)malloc(sizeof(float)*MAX_KS_BUFFER_LEN); 
}

void ks_deinit(void) {
    for(int i=0;i<AMY_KS_OSCS;i++) free(ks_buffer[i]);
    free(ks_buffer);
}
#endif
