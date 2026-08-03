/**
 * Fullscreen toggling.
 *
 * An RTS benefits from screen area more than most genres — the map is the
 * interface — so this is worth having rather than leaving to the browser's own
 * F11, which players on a Mac in particular may not reach for.
 *
 * Two things make this fiddlier than it looks:
 *
 *  - Safari still exposes only the `webkit`-prefixed API, so both spellings are
 *    probed.
 *  - `requestFullscreen` must happen inside a user gesture and rejects
 *    otherwise, so failures are reported rather than thrown. A browser that
 *    refuses (an iframe without `allow="fullscreen"`, for instance) should
 *    leave the game running normally, not break it.
 */

interface WebkitDocument extends Document {
  webkitFullscreenElement?: Element | null;
  webkitExitFullscreen?: () => Promise<void> | void;
}

interface WebkitElement extends HTMLElement {
  webkitRequestFullscreen?: () => Promise<void> | void;
}

/** True when anything is currently presented fullscreen. */
export function isFullscreen(): boolean {
  const doc = document as WebkitDocument;
  return Boolean(doc.fullscreenElement ?? doc.webkitFullscreenElement);
}

/** True when this browser offers the API at all. */
export function fullscreenSupported(): boolean {
  const el = document.documentElement as WebkitElement;
  return Boolean(el.requestFullscreen ?? el.webkitRequestFullscreen);
}

/**
 * Toggle fullscreen. Resolves to the state actually reached.
 *
 * Must be called from a user gesture (click or keypress) — browsers reject it
 * otherwise, which is why nothing here tries to enter fullscreen on load.
 */
export async function toggleFullscreen(target: HTMLElement = document.documentElement): Promise<boolean> {
  const doc = document as WebkitDocument;
  try {
    if (isFullscreen()) {
      await (doc.exitFullscreen?.() ?? doc.webkitExitFullscreen?.());
      return false;
    }
    const el = target as WebkitElement;
    await (el.requestFullscreen?.() ?? el.webkitRequestFullscreen?.());
    return true;
  } catch {
    // Denied by the browser (no gesture, or a permissions policy). Report the
    // real state rather than pretending the toggle worked.
    return isFullscreen();
  }
}

/** Subscribe to fullscreen changes, including ones the browser initiates itself. */
export function onFullscreenChange(handler: (active: boolean) => void): () => void {
  const listener = (): void => handler(isFullscreen());
  document.addEventListener('fullscreenchange', listener);
  document.addEventListener('webkitfullscreenchange', listener);
  return () => {
    document.removeEventListener('fullscreenchange', listener);
    document.removeEventListener('webkitfullscreenchange', listener);
  };
}
