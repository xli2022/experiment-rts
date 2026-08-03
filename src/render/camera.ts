/**
 * RTS camera.
 *
 * A fixed-pitch perspective camera that pans over the ground plane and zooms by
 * dollying along its view direction. Deliberately not an orbit camera: RTS
 * players navigate by muscle memory over a map they know, and a freely rotating
 * camera destroys that — "my base is up and left" has to stay true.
 *
 * Camera state is pure presentation. Nothing here touches the simulation, so two
 * players looking at different parts of the map still run identical games.
 */

import * as THREE from 'three';

const PITCH = THREE.MathUtils.degToRad(52);
const MIN_HEIGHT = 12;
const MAX_HEIGHT = 85;
const PAN_SPEED = 26; // world units per second at mid zoom
const EDGE_MARGIN_PX = 12;

export class RtsCamera {
  readonly camera: THREE.PerspectiveCamera;

  /** Point on the ground the camera looks at. */
  private readonly focus = new THREE.Vector3();
  private height = 42;

  private readonly keys = new Set<string>();
  /**
   * `known` stays false until a real pointermove arrives.
   *
   * A default of (0,0) is indistinguishable from the cursor sitting in the
   * top-left corner, so edge scrolling would fire the instant the page loads and
   * slide the camera off the player's base before they touch anything.
   */
  private pointer = { x: 0, y: 0, inside: false, known: false };
  private dragging = false;
  private lastDrag = { x: 0, y: 0 };
  /** Suppress edge scrolling while the cursor is over the HUD. */
  overUi = false;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly mapSize: number,
  ) {
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.5, 400);
    this.focus.set(mapSize * 0.25, 0, mapSize * 0.25);
    this.apply();
    this.attach();
  }

  /** Centre the view on a world position. */
  lookAt(x: number, z: number): void {
    this.focus.set(x, 0, z);
    this.apply();
  }

  get focusPoint(): THREE.Vector3 {
    return this.focus;
  }

  get zoomHeight(): number {
    return this.height;
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  }

  update(dtSeconds: number): void {
    let dx = 0;
    let dz = 0;

    // Arrow keys only — deliberately not WASD.
    //
    // A, S and H are unit commands (attack-move, stop, hold), and those letters
    // are close to sacred in this genre. Panning on the same keys meant every
    // attempt to scroll left also ordered an attack-move, which is worse than
    // having one fewer way to move the camera. Edge scroll, middle-drag and the
    // minimap all still work.
    if (this.keys.has('ArrowUp')) dz -= 1;
    if (this.keys.has('ArrowDown')) dz += 1;
    if (this.keys.has('ArrowLeft')) dx -= 1;
    if (this.keys.has('ArrowRight')) dx += 1;

    // Edge scrolling, the RTS convention. Disabled over the HUD so reaching for
    // a button does not send the camera flying.
    if (this.pointer.known && this.pointer.inside && !this.overUi && !this.dragging) {
      const rect = this.canvas.getBoundingClientRect();
      if (this.pointer.x < EDGE_MARGIN_PX) dx -= 1;
      if (this.pointer.x > rect.width - EDGE_MARGIN_PX) dx += 1;
      if (this.pointer.y < EDGE_MARGIN_PX) dz -= 1;
      if (this.pointer.y > rect.height - EDGE_MARGIN_PX) dz += 1;
    }

    if (dx !== 0 || dz !== 0) {
      const len = Math.hypot(dx, dz);
      // Pan faster when zoomed out, so crossing the map takes similar time at
      // any zoom level.
      const speed = PAN_SPEED * (this.height / 40) * dtSeconds;
      this.focus.x += (dx / len) * speed;
      this.focus.z += (dz / len) * speed;
      this.clampFocus();
      this.apply();
    }
  }

  private clampFocus(): void {
    const pad = 4;
    this.focus.x = THREE.MathUtils.clamp(this.focus.x, pad, this.mapSize - pad);
    this.focus.z = THREE.MathUtils.clamp(this.focus.z, pad, this.mapSize - pad);
  }

  private apply(): void {
    const back = Math.cos(PITCH) * this.height;
    const up = Math.sin(PITCH) * this.height;
    this.camera.position.set(this.focus.x, up, this.focus.z + back);
    this.camera.lookAt(this.focus);
  }

  private attach(): void {
    window.addEventListener('keydown', (e) => {
      // Ignore camera keys while typing in a form field.
      if (e.target instanceof HTMLInputElement) return;
      this.keys.add(e.code);
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());

    this.canvas.addEventListener('pointerenter', () => {
      this.pointer.inside = true;
    });
    this.canvas.addEventListener('pointerleave', () => {
      this.pointer.inside = false;
    });
    this.canvas.addEventListener('pointermove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.pointer.x = e.clientX - rect.left;
      this.pointer.y = e.clientY - rect.top;
      this.pointer.known = true;
      if (this.dragging) {
        const scale = this.height / 600;
        this.focus.x -= (e.clientX - this.lastDrag.x) * scale;
        this.focus.z -= (e.clientY - this.lastDrag.y) * scale;
        this.lastDrag = { x: e.clientX, y: e.clientY };
        this.clampFocus();
        this.apply();
      }
    });

    // Middle-drag pans, matching the convention players expect.
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 1) return;
      e.preventDefault();
      this.dragging = true;
      this.lastDrag = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('pointerup', (e) => {
      if (e.button === 1) this.dragging = false;
    });

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.height = THREE.MathUtils.clamp(
          this.height * (1 + Math.sign(e.deltaY) * 0.12),
          MIN_HEIGHT,
          MAX_HEIGHT,
        );
        this.apply();
      },
      { passive: false },
    );
  }
}
