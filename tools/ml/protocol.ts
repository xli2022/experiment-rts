/**
 * The wire between Python and a Bun environment process.
 *
 * Frames on stdin and stdout, each:
 *
 *   u32 length (of everything after it) | u8 kind | u32 headerLength |
 *   header JSON | payload
 *
 * little-endian. The header names the arrays in the payload, in order, with
 * their dtype and shape, so the payload is raw bytes with no framing of its
 * own — a step's observations for a dozen environments is one frame and one
 * write. `rtsml/protocol.py` is the same layout in Python.
 */

export enum Kind {
  Hello = 1,
  Reset = 2,
  Step = 3,
  Obs = 4,
  Error = 5,
  Close = 6,
}

export type DType = 'f32' | 'u8' | 'i32';

export interface ArrayDesc {
  name: string;
  dtype: DType;
  shape: number[];
}

export interface Frame<H = unknown> {
  kind: Kind;
  header: H;
  arrays: ArrayBufferView[];
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function dtypeOf(view: ArrayBufferView): DType {
  if (view instanceof Float32Array) return 'f32';
  if (view instanceof Uint8Array) return 'u8';
  if (view instanceof Int32Array) return 'i32';
  throw new Error('unsupported array type');
}

function bytesOf(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

/** Serialise a frame. `arrays` are listed in the header as `arrays: ArrayDesc[]`. */
export function encodeFrame(
  kind: Kind,
  header: Record<string, unknown>,
  arrays: readonly { name: string; view: ArrayBufferView; shape?: number[] }[] = [],
): Uint8Array {
  const descs: ArrayDesc[] = arrays.map((a) => ({
    name: a.name,
    dtype: dtypeOf(a.view),
    shape: a.shape ?? [a.view.byteLength / bytesPer(dtypeOf(a.view))],
  }));
  const headerBytes = encoder.encode(JSON.stringify({ ...header, arrays: descs }));
  let payload = 0;
  for (const a of arrays) payload += a.view.byteLength;
  const length = 1 + 4 + headerBytes.length + payload;
  const out = new Uint8Array(4 + length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, length, true);
  out[4] = kind;
  dv.setUint32(5, headerBytes.length, true);
  out.set(headerBytes, 9);
  let at = 9 + headerBytes.length;
  for (const a of arrays) {
    out.set(bytesOf(a.view), at);
    at += a.view.byteLength;
  }
  return out;
}

export function bytesPer(dtype: DType): number {
  return dtype === 'u8' ? 1 : 4;
}

function viewOf(dtype: DType, bytes: Uint8Array): ArrayBufferView {
  // Copy into a fresh, aligned buffer: a slice of the stream is not aligned.
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  switch (dtype) {
    case 'f32':
      return new Float32Array(copy.buffer);
    case 'i32':
      return new Int32Array(copy.buffer);
    case 'u8':
      return copy;
  }
}

/** Parse one complete frame from `bytes` (which must be exactly one frame). */
export function decodeFrame<H = Record<string, unknown>>(
  bytes: Uint8Array,
): Frame<H & { arrays: ArrayDesc[] }> {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = dv.getUint32(0, true);
  if (bytes.length !== 4 + length)
    throw new Error(`frame is ${bytes.length} bytes, header says ${4 + length}`);
  const kind = bytes[4] as Kind;
  const headerLength = dv.getUint32(5, true);
  const header = JSON.parse(decoder.decode(bytes.subarray(9, 9 + headerLength))) as H & {
    arrays: ArrayDesc[];
  };
  const arrays: ArrayBufferView[] = [];
  let at = 9 + headerLength;
  for (const desc of header.arrays ?? []) {
    const count = desc.shape.reduce((n, d) => n * d, 1);
    const size = count * bytesPer(desc.dtype);
    arrays.push(viewOf(desc.dtype, bytes.subarray(at, at + size)));
    at += size;
  }
  if (at !== bytes.length) throw new Error(`frame has ${bytes.length - at} trailing bytes`);
  return { kind, header, arrays };
}

/** Feeds arbitrary chunks and yields complete frames, however the bytes were split. */
export class FrameParser {
  private buffer = new Uint8Array(0);

  push(chunk: Uint8Array): Uint8Array[] {
    const joined = new Uint8Array(this.buffer.length + chunk.length);
    joined.set(this.buffer);
    joined.set(chunk, this.buffer.length);
    const frames: Uint8Array[] = [];
    let at = 0;
    while (joined.length - at >= 4) {
      const length = new DataView(joined.buffer, joined.byteOffset + at, 4).getUint32(0, true);
      if (joined.length - at < 4 + length) break;
      frames.push(joined.slice(at, at + 4 + length));
      at += 4 + length;
    }
    this.buffer = joined.slice(at);
    return frames;
  }
}
