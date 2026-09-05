import { readFile } from 'node:fs/promises';

const FBX_BINARY_MAGIC = 'Kaydara FBX Binary  \0\x1a\0';
const FBX_TIME_UNIT = 46_186_158_000;
const utf8 = new TextDecoder();

/**
 * Read the authored AnimationStack windows from an FBX without decoding its
 * geometry. FBX animation curves can extend far outside this interval; Unity
 * honors the stack window, while Three's FBXLoader exposes the whole envelope.
 */
export async function readFbxTakeWindows(path) {
  return parseFbxTakeWindows(await readFile(path));
}

export function parseFbxTakeWindows(bytes) {
  const view = asDataView(bytes);
  if (ascii(view, 0, FBX_BINARY_MAGIC.length) === FBX_BINARY_MAGIC) {
    return parseBinaryTakeWindows(view);
  }
  return parseAsciiTakeWindows(utf8.decode(view));
}

function parseBinaryTakeWindows(view) {
  const version = view.getUint32(23, true);
  const wide = version >= 7500;
  const nullRecordLength = wide ? 25 : 13;
  const state = { offset: 27 };
  const windows = [];

  while (state.offset + nullRecordLength <= view.byteLength) {
    const node = readNodeHeader(view, state, wide);
    if (!node) break;
    if (node.name === 'Objects') {
      readObjectNodes(view, state, node.endOffset, wide, windows);
    } else {
      state.offset = node.endOffset;
    }
  }
  return Object.freeze(windows.map(freezeWindow));
}

function readObjectNodes(view, state, endOffset, wide, windows) {
  while (state.offset < endOffset) {
    const node = readNodeHeader(view, state, wide);
    if (!node) break;
    if (node.name === 'AnimationStack') {
      const name = animationStackName(node.properties);
      const values = new Map();
      readAnimationStackChildren(view, state, node.endOffset, wide, values);
      const start = fbxTime(values.get('LocalStart') ?? 0);
      const stop = fbxTime(values.get('LocalStop'));
      if (Number.isFinite(start) && Number.isFinite(stop) && stop > start) {
        windows.push({ name, start, stop });
      }
    } else {
      state.offset = node.endOffset;
    }
  }
  state.offset = endOffset;
}

function readAnimationStackChildren(view, state, endOffset, wide, values) {
  while (state.offset < endOffset) {
    const node = readNodeHeader(view, state, wide);
    if (!node) break;
    if (node.name === 'Properties70') {
      readProperties70(view, state, node.endOffset, wide, values);
    } else {
      state.offset = node.endOffset;
    }
  }
  state.offset = endOffset;
}

function readProperties70(view, state, endOffset, wide, values) {
  while (state.offset < endOffset) {
    const node = readNodeHeader(view, state, wide);
    if (!node) break;
    if (node.name === 'P' && typeof node.properties[0] === 'string') {
      const value = node.properties.at(-1);
      if (typeof value === 'number' || typeof value === 'bigint') {
        values.set(node.properties[0], value);
      }
    }
    state.offset = node.endOffset;
  }
  state.offset = endOffset;
}

function readNodeHeader(view, state, wide) {
  const headerLength = wide ? 25 : 13;
  if (state.offset + headerLength > view.byteLength) return null;
  const endOffset = readOffset(view, state, wide);
  const propertyCount = readOffset(view, state, wide);
  readOffset(view, state, wide); // property-list byte length
  const nameLength = view.getUint8(state.offset++);
  if (endOffset === 0) return null;
  if (
    !Number.isSafeInteger(endOffset) ||
    endOffset > view.byteLength ||
    endOffset < state.offset + nameLength
  ) {
    throw new Error(`Invalid FBX node end offset ${endOffset}`);
  }
  const name = ascii(view, state.offset, nameLength);
  state.offset += nameLength;
  const properties = [];
  for (let index = 0; index < propertyCount; index++) {
    properties.push(readProperty(view, state));
  }
  return { endOffset, name, properties };
}

function readOffset(view, state, wide) {
  if (wide) {
    const value = view.getBigUint64(state.offset, true);
    state.offset += 8;
    const number = Number(value);
    if (!Number.isSafeInteger(number)) {
      throw new Error(`FBX offset exceeds JavaScript's safe integer range`);
    }
    return number;
  }
  const value = view.getUint32(state.offset, true);
  state.offset += 4;
  return value;
}

function readProperty(view, state) {
  const type = String.fromCharCode(view.getUint8(state.offset++));
  switch (type) {
    case 'Y': {
      const value = view.getInt16(state.offset, true);
      state.offset += 2;
      return value;
    }
    case 'C':
      return view.getUint8(state.offset++) !== 0;
    case 'I': {
      const value = view.getInt32(state.offset, true);
      state.offset += 4;
      return value;
    }
    case 'F': {
      const value = view.getFloat32(state.offset, true);
      state.offset += 4;
      return value;
    }
    case 'D': {
      const value = view.getFloat64(state.offset, true);
      state.offset += 8;
      return value;
    }
    case 'L': {
      const value = view.getBigInt64(state.offset, true);
      state.offset += 8;
      return value;
    }
    case 'S':
    case 'R': {
      const length = view.getUint32(state.offset, true);
      state.offset += 4;
      const start = state.offset;
      state.offset += length;
      return type === 'S' ? utf8.decode(slice(view, start, length)) : null;
    }
    case 'b':
    case 'c':
    case 'd':
    case 'f':
    case 'i':
    case 'l': {
      state.offset += 8; // element count and compression encoding
      const byteLength = view.getUint32(state.offset, true);
      state.offset += 4 + byteLength;
      return null;
    }
    default:
      throw new Error(`Unsupported FBX property type ${JSON.stringify(type)}`);
  }
}

function parseAsciiTakeWindows(view) {
  const windows = [];
  const startPattern = /^\s*AnimationStack:\s*[^,]+,\s*"(?:AnimStack::)?([^"]*)"/gm;
  for (const match of view.matchAll(startPattern)) {
    const block = braceBlock(view, match.index);
    const start = asciiTime(block, 'LocalStart') ?? 0;
    const stop = asciiTime(block, 'LocalStop');
    if (Number.isFinite(start) && Number.isFinite(stop) && stop > start) {
      windows.push({
        name: match[1],
        start: fbxTime(start),
        stop: fbxTime(stop),
      });
    }
  }
  return Object.freeze(windows.map(freezeWindow));
}

function braceBlock(source, start) {
  const opening = source.indexOf('{', start);
  if (opening < 0) return '';
  let depth = 0;
  for (let index = opening; index < source.length; index++) {
    if (source[index] === '{') depth++;
    if (source[index] === '}' && --depth === 0) {
      return source.slice(opening + 1, index);
    }
  }
  return source.slice(opening + 1);
}

function asciiTime(block, property) {
  const escaped = property.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(
    `^\\s*P:\\s*"${escaped}"[^\\r\\n,]*(?:,[^\\r\\n,]*){3},\\s*(-?\\d+)`,
    'm',
  ).exec(block);
  return match ? Number(match[1]) : null;
}

function animationStackName(properties) {
  const raw = properties.find((value) => typeof value === 'string') ?? '';
  return raw.split('\0', 1)[0].replace(/^AnimStack::/, '');
}

function fbxTime(value) {
  return Number(value) / FBX_TIME_UNIT;
}

function freezeWindow(window) {
  return Object.freeze(window);
}

function asDataView(bytes) {
  if (bytes instanceof DataView) return bytes;
  if (bytes instanceof ArrayBuffer) return new DataView(bytes);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function ascii(view, offset, length) {
  return String.fromCharCode(...slice(view, offset, length));
}

function slice(view, offset, length) {
  return new Uint8Array(view.buffer, view.byteOffset + offset, length);
}
