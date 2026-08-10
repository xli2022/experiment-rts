/**
 * On-demand viewer for every authored unit model.
 *
 * The cards are DOM, but their previews are all drawn through the app's one
 * WebGL renderer. Scissoring the existing canvas avoids both a WebGL context per
 * card and a second full-size canvas while still allowing an ordinary scrolling,
 * accessible grid.
 */

import * as THREE from "three";
import { KTX2Loader } from "three/examples/jsm/loaders/KTX2Loader.js";
import { AnimatedUnitPool } from "./animatedUnits.js";
import { loadAnimatedModel, type AnimatedModel } from "./models/animated.js";

const LOAD_CONCURRENCY = 4;
// The original shared scale used 1.7; 2.55 keeps every proportion intact while
// making each preview 50% larger.
const PREVIEW_LARGEST_MODEL_SIZE = 2.55;
const PREVIEW_VIEW_HEIGHT = 3.6;
// Keep the surrounding gallery in the game's dark theme, but render every
// model against a light neutral card so dark silhouettes stay legible.
const GALLERY_CLEAR = 0x0c131d;
const CARD_CLEAR = 0xedf1f5;
const FACTION_ORDER = ["Human", "Robot", "Monster", "Undead"] as const;
type UnitFaction = (typeof FACTION_ORDER)[number];

interface CatalogModel {
  unit: string;
  faction: UnitFaction;
  file: string;
  skins: [string, string];
  /** Athena2's first baked run frame, in its shared world scale. */
  runSize: [number, number, number];
  /** Desired y=0 plane in the final GLB's model-space run coordinates. */
  runGroundY?: number;
}

interface GalleryPreview {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  pool: AnimatedUnitPool;
  model: AnimatedModel;
  texture: THREE.Texture | null;
  guide: THREE.GridHelper;
  matrix: THREE.Matrix4;
}

interface LoadSession {
  cancelled: boolean;
  abort: AbortController;
  completed: number;
  failed: number;
  untextured: number;
  total: number;
}

/**
 * Modal model gallery. Constructing it does not touch the DOM or load assets;
 * everything is deferred until `open`, and everything GPU-owned is released by
 * `close`.
 */
export class UnitGallery {
  private overlay: HTMLElement | null = null;
  private scroll: HTMLElement | null = null;
  private groups: HTMLElement | null = null;
  private status: HTMLElement | null = null;
  private closeButton: HTMLButtonElement | null = null;
  private returnFocus: HTMLElement | null = null;
  private observer: IntersectionObserver | null = null;
  private session: LoadSession | null = null;

  private readonly cards: HTMLElement[] = [];
  private readonly slots: HTMLElement[] = [];
  private readonly visible = new Set<number>();
  private readonly previews = new Map<number, GalleryPreview>();

  private readonly priorClear = new THREE.Color();
  private readonly priorViewport = new THREE.Vector4();
  private readonly priorScissor = new THREE.Vector4();

  constructor(
    private readonly root: HTMLElement,
    private readonly renderer: THREE.WebGLRenderer,
  ) {}

  get isOpen(): boolean {
    return this.overlay !== null;
  }

  /** Show the shell immediately, then progressively fill it with models. */
  open(): void {
    if (this.isOpen) return;

    this.returnFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    const overlay = document.createElement("div");
    overlay.className = "unit-gallery-overlay interactive";
    overlay.innerHTML = `
      <section class="unit-gallery-dialog" id="unit-gallery-dialog"
               role="dialog" aria-modal="true"
               aria-labelledby="unit-gallery-title"
               aria-describedby="unit-gallery-description" tabindex="-1">
        <header class="unit-gallery-header">
          <div>
            <h1 id="unit-gallery-title">All units</h1>
            <p id="unit-gallery-description">Every authored model at a shared scale, grouped by faction and looping its run animation.</p>
            <p class="unit-gallery-status" role="status" aria-live="polite">Loading unit list…</p>
          </div>
          <button class="unit-gallery-close" type="button" aria-label="Close all units">Close</button>
        </header>
        <div class="unit-gallery-scroll">
          <div class="unit-gallery-groups" aria-busy="true"></div>
        </div>
      </section>
    `;
    this.root.append(overlay);
    this.root.classList.add("unit-gallery-open");

    this.overlay = overlay;
    required(overlay, ".unit-gallery-dialog");
    this.scroll = required(overlay, ".unit-gallery-scroll");
    this.groups = required(overlay, ".unit-gallery-groups");
    this.status = required(overlay, ".unit-gallery-status");
    this.closeButton = required(
      overlay,
      ".unit-gallery-close",
    ) as HTMLButtonElement;

    this.closeButton.addEventListener("click", () => this.close());
    overlay.addEventListener("pointerdown", (event) => {
      if (event.target === overlay) this.close();
    });
    overlay.addEventListener("keydown", this.handleModalKey);
    this.closeButton.focus();

    const session: LoadSession = {
      cancelled: false,
      abort: new AbortController(),
      completed: 0,
      failed: 0,
      untextured: 0,
      total: 0,
    };
    this.session = session;
    void this.load(session);
  }

  /**
   * Draw every visible, loaded card. Call instead of the normal game-scene
   * render while `isOpen`; simulation can continue independently.
   */
  render(elapsedSeconds: number): void {
    if (!this.overlay || !this.scroll) return;

    const renderer = this.renderer;
    const canvasRect = renderer.domElement.getBoundingClientRect();
    const scrollRect = this.scroll.getBoundingClientRect();
    if (canvasRect.width <= 0 || canvasRect.height <= 0) return;

    const priorAlpha = renderer.getClearAlpha();
    renderer.getClearColor(this.priorClear);
    renderer.getViewport(this.priorViewport);
    renderer.getScissor(this.priorScissor);
    const priorScissorTest = renderer.getScissorTest();
    const priorAutoClear = renderer.autoClear;

    try {
      renderer.autoClear = false;
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, canvasRect.width, canvasRect.height);
      renderer.setClearColor(GALLERY_CLEAR, 1);
      renderer.clear(true, true, true);

      const indices = this.observer ? this.visible : this.previews.keys();
      for (const index of indices) {
        const preview = this.previews.get(index);
        const slot = this.slots[index];
        if (!preview || !slot?.isConnected) continue;

        const rect = slot.getBoundingClientRect();
        const clipLeft = Math.max(rect.left, scrollRect.left, canvasRect.left);
        const clipRight = Math.min(
          rect.right,
          scrollRect.right,
          canvasRect.right,
        );
        const clipTop = Math.max(rect.top, scrollRect.top, canvasRect.top);
        const clipBottom = Math.min(
          rect.bottom,
          scrollRect.bottom,
          canvasRect.bottom,
        );
        if (clipRight <= clipLeft || clipBottom <= clipTop) continue;

        const viewportX = rect.left - canvasRect.left;
        const viewportY = canvasRect.bottom - rect.bottom;
        const scissorX = clipLeft - canvasRect.left;
        const scissorY = canvasRect.bottom - clipBottom;

        renderer.setViewport(viewportX, viewportY, rect.width, rect.height);
        renderer.setScissor(
          scissorX,
          scissorY,
          clipRight - clipLeft,
          clipBottom - clipTop,
        );
        renderer.setScissorTest(true);
        renderer.setClearColor(CARD_CLEAR, 1);
        renderer.clear(true, true, true);

        const aspect = rect.width / Math.max(1, rect.height);
        const halfHeight = PREVIEW_VIEW_HEIGHT / 2;
        preview.camera.left = -halfHeight * aspect;
        preview.camera.right = halfHeight * aspect;
        preview.camera.top = halfHeight;
        preview.camera.bottom = -halfHeight;
        preview.camera.updateProjectionMatrix();

        const frame = AnimatedUnitPool.framePairFor(
          preview.model,
          "run",
          elapsedSeconds,
          true,
        );
        preview.pool.begin();
        preview.pool.add(preview.matrix, frame.from, frame.to, frame.blend);
        preview.pool.commit();
        renderer.render(preview.scene, preview.camera);
      }
    } finally {
      renderer.autoClear = priorAutoClear;
      renderer.setViewport(this.priorViewport);
      renderer.setScissor(this.priorScissor);
      renderer.setScissorTest(priorScissorTest);
      renderer.setClearColor(this.priorClear, priorAlpha);
    }
  }

  /** Close the modal, stop scheduling loads, and release every owned resource. */
  close(): void {
    this.root.classList.remove("unit-gallery-open");
    if (!this.overlay) return;

    const session = this.session;
    if (session) {
      session.cancelled = true;
      session.abort.abort();
      this.session = null;
    }

    this.observer?.disconnect();
    this.observer = null;
    for (const preview of this.previews.values()) disposePreview(preview);
    this.previews.clear();
    this.visible.clear();
    this.cards.length = 0;
    this.slots.length = 0;

    this.overlay.removeEventListener("keydown", this.handleModalKey);
    this.overlay.remove();
    this.overlay = null;
    this.scroll = null;
    this.groups = null;
    this.status = null;
    this.closeButton = null;

    const focus = this.returnFocus;
    this.returnFocus = null;
    if (focus?.isConnected) focus.focus();
  }

  dispose(): void {
    this.close();
  }

  private readonly handleModalKey = (event: KeyboardEvent): void => {
    // The modal is presentation-only: none of its keystrokes may issue commands
    // to the match underneath it.
    event.stopPropagation();
    if (event.key === "Escape") {
      event.preventDefault();
      this.close();
      return;
    }
    if (event.key === "Tab" && this.closeButton) {
      // Close is deliberately the only control, so keeping focus on it is the
      // complete focus trap rather than a special case.
      event.preventDefault();
      this.closeButton.focus();
    }
  };

  private async load(session: LoadSession): Promise<void> {
    try {
      const response = await fetch(modelAssetUrl("all-units.json"), {
        signal: session.abort.signal,
      });
      if (!response.ok)
        throw new Error(`unit list returned ${response.status}`);
      const entries = parseCatalog(await response.json());
      if (session.cancelled) return;
      const largestRunExtent = Math.max(
        ...entries.flatMap((entry) => entry.runSize),
      );
      const worldScale = PREVIEW_LARGEST_MODEL_SIZE / largestRunExtent;

      session.total = entries.length;
      this.mountCards(entries);
      this.updateProgress(session);

      const skinLoader = new KTX2Loader().detectSupport(this.renderer);
      let next = 0;
      const worker = async (): Promise<void> => {
        while (!session.cancelled) {
          const index = next++;
          const entry = entries[index];
          if (!entry) return;
          await this.loadOne(session, skinLoader, entry, index, worldScale);
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(LOAD_CONCURRENCY, entries.length) }, () =>
          worker(),
        ),
      );
      skinLoader.dispose();

      if (!session.cancelled && this.groups) {
        this.groups.setAttribute("aria-busy", "false");
        this.updateProgress(session);
      }
    } catch (error) {
      if (session.cancelled || isAbort(error)) return;
      if (this.status)
        this.status.textContent = "Unit models could not be loaded.";
      if (this.groups) {
        this.groups.setAttribute("aria-busy", "false");
        const message = document.createElement("p");
        message.className = "unit-gallery-catalog-error";
        message.textContent =
          error instanceof Error ? error.message : String(error);
        this.groups.replaceChildren(message);
      }
    }
  }

  private mountCards(entries: readonly CatalogModel[]): void {
    if (!this.groups || !this.scroll) return;
    this.groups.replaceChildren();

    this.observer?.disconnect();
    this.observer =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(
            (changes) => {
              for (const change of changes) {
                const index = Number(
                  (change.target as HTMLElement).dataset.galleryIndex,
                );
                if (change.isIntersecting) this.visible.add(index);
                else this.visible.delete(index);
              }
            },
            { root: this.scroll, rootMargin: "80px" },
          );

    const factionGrids = new Map<UnitFaction, HTMLElement>();
    for (const faction of FACTION_ORDER) {
      const section = document.createElement("section");
      section.className = "unit-gallery-faction";
      const heading = document.createElement("h2");
      const headingId = `unit-gallery-faction-${faction.toLowerCase()}`;
      heading.id = headingId;
      heading.className = "unit-gallery-faction-title";
      heading.textContent = faction;

      const count = document.createElement("span");
      const factionCount = entries.filter(
        (entry) => entry.faction === faction,
      ).length;
      count.textContent = `${factionCount} units`;
      heading.append(count);

      const grid = document.createElement("div");
      grid.className = "unit-gallery-grid";
      grid.setAttribute("role", "list");
      section.setAttribute("aria-labelledby", headingId);
      section.append(heading, grid);
      this.groups.append(section);
      factionGrids.set(faction, grid);
    }

    entries.forEach((entry, index) => {
      const card = document.createElement("article");
      card.className = "unit-gallery-card";
      card.setAttribute("role", "listitem");
      card.setAttribute(
        "aria-label",
        `${entry.unit}, ${entry.faction} faction model preview`,
      );

      const slot = document.createElement("div");
      slot.className = "unit-gallery-preview";
      slot.dataset.galleryIndex = String(index);
      slot.setAttribute("aria-hidden", "true");

      const placeholder = document.createElement("span");
      placeholder.className = "unit-gallery-placeholder";
      placeholder.textContent = "Loading…";
      slot.append(placeholder);

      const label = document.createElement("div");
      label.className = "unit-gallery-label";
      label.textContent = entry.unit;
      card.append(slot, label);
      factionGrids.get(entry.faction)!.append(card);
      this.cards.push(card);
      this.slots.push(slot);
      this.observer?.observe(slot);
    });
  }

  private async loadOne(
    session: LoadSession,
    skinLoader: KTX2Loader,
    entry: CatalogModel,
    index: number,
    worldScale: number,
  ): Promise<void> {
    let model: AnimatedModel | null = null;
    let texture: THREE.Texture | null = null;
    try {
      model = await loadAnimatedModel(modelAssetUrl(entry.file), "run");
      if (session.cancelled) return;

      try {
        texture = await skinLoader.loadAsync(modelAssetUrl(entry.skins[0]));
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.needsUpdate = true;
      } catch {
        session.untextured++;
      }
      if (session.cancelled) return;

      const preview = createPreview(
        model,
        texture,
        entry.runSize,
        worldScale,
        entry.runGroundY,
      );
      this.previews.set(index, preview);
      model = null;
      texture = null;
      this.cards[index]?.classList.add("loaded");
    } catch (error) {
      session.failed++;
      if (!session.cancelled) {
        const card = this.cards[index];
        if (card) {
          card.classList.add("error");
          const placeholder = card.querySelector(".unit-gallery-placeholder");
          if (placeholder) placeholder.textContent = "Unavailable";
          card.title = error instanceof Error ? error.message : String(error);
        }
      }
    } finally {
      if (model) disposeModel(model);
      texture?.dispose();
      session.completed++;
      if (!session.cancelled) this.updateProgress(session);
    }
  }

  private updateProgress(session: LoadSession): void {
    if (!this.status) return;
    if (session.total === 0) {
      this.status.textContent = "Loading unit list…";
      return;
    }
    if (session.completed < session.total) {
      this.status.textContent = `Loading models ${session.completed} of ${session.total}…`;
      return;
    }

    const ready = session.total - session.failed;
    const notes: string[] = [`${ready} models ready`];
    if (session.failed > 0) notes.push(`${session.failed} unavailable`);
    if (session.untextured > 0)
      notes.push(`${session.untextured} without team skin`);
    this.status.textContent = notes.join(" · ");
  }
}

function createPreview(
  model: AnimatedModel,
  texture: THREE.Texture | null,
  authoredRunSize: readonly [number, number, number],
  worldScale: number,
  runGroundY?: number,
): GalleryPreview {
  const size = model.firstFrameBounds.getSize(new THREE.Vector3());
  const modelExtent = Math.max(0.001, size.x, size.y, size.z);
  const authoredExtent = Math.max(...authoredRunSize);
  const scale = proportionalPreviewScale(
    authoredExtent,
    modelExtent,
    worldScale,
  );
  const center = model.animatedBounds.getCenter(new THREE.Vector3());
  const groundOffset = previewGroundOffset(
    runGroundY,
    model.animatedBounds.min.y,
    scale,
  );

  const matrix = new THREE.Matrix4().compose(
    new THREE.Vector3(-center.x * scale, groundOffset, -center.z * scale),
    new THREE.Quaternion(),
    new THREE.Vector3(scale, scale, scale),
  );

  const material = new THREE.MeshLambertMaterial({
    map: texture,
    color: texture ? 0xffffff : 0x4a9eff,
  });
  const pool = new AnimatedUnitPool(model, material, 1);
  pool.mesh.castShadow = false;

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xaeb9c5, 2.25));
  const sun = new THREE.DirectionalLight(0xffedda, 2.35);
  sun.position.set(3, 5, 4);
  const fill = new THREE.DirectionalLight(0xbddcff, 1.15);
  fill.position.set(-4, 2.5, -3);
  scene.add(sun, fill, pool.mesh);

  // Two Athena2 world units in every card: the identical guide and camera make
  // the size difference between a Parasite and a dragon visible at a glance.
  const guide = new THREE.GridHelper(2 * worldScale, 2, 0x778597, 0xaeb8c4);
  const guideMaterials = Array.isArray(guide.material)
    ? guide.material
    : [guide.material];
  for (const guideMaterial of guideMaterials) {
    guideMaterial.transparent = true;
    guideMaterial.opacity = 0.55;
    guideMaterial.depthWrite = false;
  }
  scene.add(guide);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
  // Keep one projection and one world scale, but translate the view vertically
  // to the centre of the grounded run envelope. This prevents tall or strongly
  // pitched units such as Fire Dragon from being clipped while small units
  // retain their authored size.
  const runCenterY = center.y * scale + groundOffset;
  camera.position.set(2.8, runCenterY + 1.3, 3.6);
  camera.lookAt(0, runCenterY, 0);

  return { scene, camera, pool, model, texture, guide, matrix };
}

/** Convert one model's source units without erasing its authored relative size. */
export function proportionalPreviewScale(
  authoredExtent: number,
  modelExtent: number,
  worldScale: number,
): number {
  return (authoredExtent * worldScale) / Math.max(0.001, modelExtent);
}

/** Translate a final-GLB ground plane to the grid, retaining the old fallback. */
export function previewGroundOffset(
  runGroundY: number | undefined,
  animatedMinY: number,
  scale: number,
): number {
  return -(runGroundY ?? animatedMinY) * scale;
}

function disposePreview(preview: GalleryPreview): void {
  preview.scene.clear();
  preview.pool.dispose();
  preview.guide.geometry.dispose();
  const materials = Array.isArray(preview.guide.material)
    ? preview.guide.material
    : [preview.guide.material];
  for (const material of materials) material.dispose();
  preview.texture?.dispose();
  disposeModel(preview.model);
}

function disposeModel(model: AnimatedModel): void {
  model.geometry.dispose();
  model.boneTexture.dispose();
}

function modelAssetUrl(file: string): string {
  return `${import.meta.env.BASE_URL}models/${file}`;
}

function parseCatalog(value: unknown): CatalogModel[] {
  if (!isRecord(value) || value.version !== 2 || !Array.isArray(value.models)) {
    throw new Error("all-units.json is not a version 2 model catalog");
  }

  return value.models.map((candidate, index) => {
    if (
      !isRecord(candidate) ||
      typeof candidate.unit !== "string" ||
      !isFaction(candidate.faction) ||
      typeof candidate.file !== "string" ||
      !Array.isArray(candidate.skins) ||
      typeof candidate.skins[0] !== "string" ||
      typeof candidate.skins[1] !== "string" ||
      !Array.isArray(candidate.runSize) ||
      candidate.runSize.length !== 3 ||
      !candidate.runSize.every(
        (size) => typeof size === "number" && Number.isFinite(size) && size > 0,
      ) ||
      (candidate.runGroundY !== undefined &&
        (typeof candidate.runGroundY !== "number" ||
          !Number.isFinite(candidate.runGroundY)))
    ) {
      throw new Error(`all-units.json model ${index} is malformed`);
    }
    return {
      unit: candidate.unit,
      faction: candidate.faction,
      file: candidate.file,
      skins: [candidate.skins[0], candidate.skins[1]],
      runSize: [
        candidate.runSize[0],
        candidate.runSize[1],
        candidate.runSize[2],
      ],
      ...(candidate.runGroundY === undefined
        ? {}
        : { runGroundY: candidate.runGroundY }),
    };
  });
}

function isFaction(value: unknown): value is UnitFaction {
  return FACTION_ORDER.some((faction) => faction === value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function required(root: HTMLElement, selector: string): HTMLElement {
  const element = root.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`unit gallery element missing: ${selector}`);
  }
  return element;
}
