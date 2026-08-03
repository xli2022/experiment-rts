/**
 * Sound effects and music, synthesised at runtime.
 *
 * Every sound here is generated with oscillators and noise buffers rather than
 * loaded from files. That keeps the repository self-contained — no binary
 * assets, no licences to track, nothing extra to download — and a handful of
 * WebAudio nodes costs far less than the samples would.
 *
 * ## Autoplay
 *
 * Browsers refuse to start an AudioContext outside a user gesture, so nothing
 * here runs until `resume()` is called from a click. Until then every method is
 * a no-op, and a browser with no WebAudio at all simply stays silent rather than
 * breaking the game.
 *
 * ## Not part of the simulation
 *
 * Sounds are triggered from the same per-tick event lists the visual effects
 * read. A muted client and a loud one run identical simulations.
 */

export type SoundName =
  | 'shot'
  | 'explosion'
  | 'death'
  | 'select'
  | 'order'
  | 'build'
  | 'ready'
  | 'denied';

/** Simultaneous voices. Beyond this, new sounds are dropped rather than queued. */
const MAX_VOICES = 24;

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private sfxBus: GainNode | null = null;
  private musicBus: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;

  private voices = 0;
  private musicTimer: number | null = null;
  private musicStep = 0;

  private _muted = false;
  private _musicEnabled = true;

  get muted(): boolean {
    return this._muted;
  }

  get ready(): boolean {
    return this.ctx !== null && this.ctx.state === 'running';
  }

  /**
   * Start (or resume) audio. Must be called from a user gesture.
   *
   * Safe to call repeatedly; browsers also suspend the context when a tab is
   * backgrounded, so this doubles as the way back.
   */
  async resume(): Promise<void> {
    try {
      if (!this.ctx) this.init();
      if (!this.ctx) return;
      if (this.ctx.state === 'suspended') await this.ctx.resume();
      if (this._musicEnabled) this.startMusic();
    } catch {
      // No WebAudio, or the gesture was not accepted. Stay silent.
    }
  }

  private init(): void {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;

    this.ctx = new Ctor();
    this.master = this.ctx.createGain();
    this.master.gain.value = this._muted ? 0 : 0.85;
    this.master.connect(this.ctx.destination);

    this.sfxBus = this.ctx.createGain();
    this.sfxBus.gain.value = 0.55;
    this.sfxBus.connect(this.master);

    this.musicBus = this.ctx.createGain();
    // Music sits well under the effects — it is atmosphere, not the signal a
    // player is listening for.
    this.musicBus.gain.value = 0.12;
    this.musicBus.connect(this.master);

    this.noiseBuffer = this.makeNoise();
  }

  /** One second of white noise, reused by every percussive sound. */
  private makeNoise(): AudioBuffer | null {
    if (!this.ctx) return null;
    const rate = this.ctx.sampleRate;
    const buffer = this.ctx.createBuffer(1, rate, rate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    return buffer;
  }

  setMuted(muted: boolean): void {
    this._muted = muted;
    if (this.master && this.ctx) {
      this.master.gain.setTargetAtTime(muted ? 0 : 0.85, this.ctx.currentTime, 0.02);
    }
  }

  toggleMuted(): boolean {
    this.setMuted(!this._muted);
    return this._muted;
  }

  /**
   * Play a sound.
   *
   * `intensity` scales volume, so a distant shot in a big battle does not hit as
   * hard as the one next to the camera. Voices are capped: a hundred simultaneous
   * shots would otherwise clip into noise and cost real CPU.
   */
  play(name: SoundName, intensity = 1): void {
    if (!this.ctx || !this.sfxBus || this.ctx.state !== 'running') return;
    if (this._muted) return;
    if (this.voices >= MAX_VOICES) return;

    const t = this.ctx.currentTime;
    const gain = this.ctx.createGain();
    gain.connect(this.sfxBus);

    this.voices++;
    // Voice accounting is approximate: release the slot slightly after the
    // longest tail so a burst cannot leak the counter upward.
    window.setTimeout(() => {
      this.voices = Math.max(0, this.voices - 1);
    }, 900);

    switch (name) {
      case 'shot':
        this.blip(gain, t, 620, 240, 0.07, 0.22 * intensity, 'square');
        break;
      case 'explosion':
      case 'death':
        this.burst(gain, t, 0.42, 0.5 * intensity, name === 'death' ? 900 : 1500);
        this.blip(gain, t, 160, 48, 0.34, 0.3 * intensity, 'sawtooth');
        break;
      case 'select':
        this.blip(gain, t, 880, 880, 0.06, 0.16 * intensity, 'sine');
        break;
      case 'order':
        this.blip(gain, t, 420, 700, 0.09, 0.16 * intensity, 'triangle');
        break;
      case 'build':
        this.blip(gain, t, 300, 520, 0.16, 0.2 * intensity, 'triangle');
        this.blip(gain, t + 0.12, 520, 760, 0.16, 0.16 * intensity, 'triangle');
        break;
      case 'ready':
        // A rising two-note chime, the classic "unit complete" cue.
        this.blip(gain, t, 660, 660, 0.11, 0.2 * intensity, 'sine');
        this.blip(gain, t + 0.1, 990, 990, 0.16, 0.18 * intensity, 'sine');
        break;
      case 'denied':
        this.blip(gain, t, 220, 160, 0.16, 0.2 * intensity, 'square');
        break;
    }
  }

  /** A pitched tone that glides from `from` to `to`. */
  private blip(
    dest: GainNode,
    when: number,
    from: number,
    to: number,
    duration: number,
    level: number,
    type: OscillatorType,
  ): void {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const env = this.ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(from, when);
    if (to !== from) osc.frequency.exponentialRampToValueAtTime(Math.max(1, to), when + duration);

    // Quick attack, exponential decay — percussive without a click at onset.
    env.gain.setValueAtTime(0.0001, when);
    env.gain.exponentialRampToValueAtTime(Math.max(0.0001, level), when + 0.008);
    env.gain.exponentialRampToValueAtTime(0.0001, when + duration);

    osc.connect(env);
    env.connect(dest);
    osc.start(when);
    osc.stop(when + duration + 0.02);
  }

  /** Filtered noise, for impacts and explosions. */
  private burst(
    dest: GainNode,
    when: number,
    duration: number,
    level: number,
    cutoff: number,
  ): void {
    if (!this.ctx || !this.noiseBuffer) return;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(cutoff, when);
    filter.frequency.exponentialRampToValueAtTime(120, when + duration);

    const env = this.ctx.createGain();
    env.gain.setValueAtTime(Math.max(0.0001, level), when);
    env.gain.exponentialRampToValueAtTime(0.0001, when + duration);

    src.connect(filter);
    filter.connect(env);
    env.connect(dest);
    src.start(when);
    src.stop(when + duration + 0.02);
  }

  // -------------------------------------------------------------------------
  // Music
  // -------------------------------------------------------------------------

  setMusicEnabled(enabled: boolean): void {
    this._musicEnabled = enabled;
    if (enabled) this.startMusic();
    else this.stopMusic();
  }

  /**
   * A slow, sparse ambient loop.
   *
   * Deliberately minimal: an RTS is listened to for tens of minutes at a stretch,
   * and anything with a strong hook becomes irritating long before the match
   * ends. A drifting minor pad plus an occasional bass note gives the silence
   * some shape without competing with the sound effects.
   */
  private startMusic(): void {
    if (!this.ctx || !this.musicBus || this.musicTimer !== null) return;

    // A natural minor scale in Hz, low in the register to stay out of the way.
    const scale = [110, 123.47, 130.81, 146.83, 164.81, 174.61, 196.0];
    const chordEvery = 4;

    const step = (): void => {
      if (!this.ctx || !this.musicBus || this._muted) return;
      const t = this.ctx.currentTime;
      const root = scale[this.musicStep % scale.length]!;

      // A sustained pad note.
      this.pad(this.musicBus, t, root, 5.5, 0.5);
      // A third above it, every few steps, for a little movement.
      if (this.musicStep % chordEvery === 0) {
        this.pad(this.musicBus, t + 0.25, root * 1.5, 4.5, 0.32);
      }
      if (this.musicStep % (chordEvery * 2) === 3) {
        this.pad(this.musicBus, t, root * 0.5, 6, 0.4);
      }
      this.musicStep++;
    };

    step();
    this.musicTimer = window.setInterval(step, 4200);
  }

  private stopMusic(): void {
    if (this.musicTimer !== null) {
      window.clearInterval(this.musicTimer);
      this.musicTimer = null;
    }
  }

  /** A soft, slowly-swelling tone. */
  private pad(dest: GainNode, when: number, freq: number, duration: number, level: number): void {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const env = this.ctx.createGain();
    const filter = this.ctx.createBiquadFilter();

    osc.type = 'triangle';
    osc.frequency.value = freq;
    osc2.type = 'sine';
    // Slight detune gives the pad some width without a chorus effect.
    osc2.frequency.value = freq * 1.005;

    filter.type = 'lowpass';
    filter.frequency.value = 700;

    // Long fade in and out, so notes bleed into each other.
    env.gain.setValueAtTime(0.0001, when);
    env.gain.exponentialRampToValueAtTime(level, when + duration * 0.4);
    env.gain.exponentialRampToValueAtTime(0.0001, when + duration);

    osc.connect(filter);
    osc2.connect(filter);
    filter.connect(env);
    env.connect(dest);
    osc.start(when);
    osc2.start(when);
    osc.stop(when + duration + 0.05);
    osc2.stop(when + duration + 0.05);
  }

  dispose(): void {
    this.stopMusic();
    void this.ctx?.close();
    this.ctx = null;
  }
}

/** The one audio engine. Sound is global; there is no reason for two. */
export const audio = new AudioEngine();
