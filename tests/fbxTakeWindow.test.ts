import { describe, expect, it } from 'vitest';

const FBX_TIME_UNIT = 46_186_158_000n;
const parser = (await import(new URL('../scripts/fbx-take-window.mjs', import.meta.url).href)) as {
  parseFbxTakeWindows(bytes: Uint8Array): readonly {
    name: string;
    start: number;
    stop: number;
  }[];
};

describe('FBX animation take windows', () => {
  it('reads an ASCII AnimationStack without accepting template defaults', () => {
    const fbx = new TextEncoder().encode(`
Definitions:  {
  ObjectType: "AnimationStack" {
    PropertyTemplate: "FbxAnimStack" {
      P: "LocalStart", "KTime", "Time", "",0
      P: "LocalStop", "KTime", "Time", "",0
    }
  }
}
Objects:  {
  AnimationStack: 1, "AnimStack::Walk", "" {
    Properties70:  {
      P: "LocalStart", "KTime", "Time", "",3079077200
      P: "LocalStop", "KTime", "Time", "",30790772000
    }
  }
}
`);

    expect(parser.parseFbxTakeWindows(fbx)).toEqual([{ name: 'Walk', start: 1 / 15, stop: 2 / 3 }]);
  });

  it('reads the same window from binary FBX nodes', () => {
    const header = Buffer.alloc(27);
    header.write('Kaydara FBX Binary  \0\x1a\0', 0, 'binary');
    header.writeUInt32LE(7400, 23);
    const startTicks = FBX_TIME_UNIT / 15n;
    const stopTicks = (FBX_TIME_UNIT * 2n) / 3n;
    const objects = binaryNode(
      'Objects',
      [],
      [
        {
          name: 'AnimationStack',
          properties: [1n, 'AnimStack::Walk', ''],
          children: [
            {
              name: 'Properties70',
              properties: [],
              children: [
                propertyNode('LocalStart', startTicks),
                propertyNode('LocalStop', stopTicks),
              ],
            },
          ],
        },
      ],
      header.length,
    );
    const fbx = Buffer.concat([header, objects, Buffer.alloc(13)]);

    expect(parser.parseFbxTakeWindows(fbx)).toEqual([{ name: 'Walk', start: 1 / 15, stop: 2 / 3 }]);
  });
});

interface BinaryNode {
  name: string;
  properties: (bigint | string)[];
  children?: BinaryNode[];
}

function propertyNode(name: string, value: bigint): BinaryNode {
  return {
    name: 'P',
    properties: [name, 'KTime', 'Time', '', value],
  };
}

function binaryNode(
  name: string,
  properties: (bigint | string)[],
  children: BinaryNode[],
  absoluteOffset: number,
): Buffer {
  const encodedName = Buffer.from(name);
  const encodedProperties = Buffer.concat(properties.map(binaryProperty));
  const headerLength = 13 + encodedName.length + encodedProperties.length;
  const encodedChildren: Buffer[] = [];
  let childOffset = absoluteOffset + headerLength;
  for (const child of children) {
    const encoded = binaryNode(child.name, child.properties, child.children ?? [], childOffset);
    encodedChildren.push(encoded);
    childOffset += encoded.length;
  }
  const terminator = children.length > 0 ? Buffer.alloc(13) : Buffer.alloc(0);
  const totalLength =
    headerLength +
    encodedChildren.reduce((sum, child) => sum + child.length, 0) +
    terminator.length;
  const header = Buffer.alloc(13);
  header.writeUInt32LE(absoluteOffset + totalLength, 0);
  header.writeUInt32LE(properties.length, 4);
  header.writeUInt32LE(encodedProperties.length, 8);
  header.writeUInt8(encodedName.length, 12);
  return Buffer.concat([header, encodedName, encodedProperties, ...encodedChildren, terminator]);
}

function binaryProperty(value: bigint | string): Buffer {
  if (typeof value === 'bigint') {
    const property = Buffer.alloc(9);
    property.write('L', 0);
    property.writeBigInt64LE(value, 1);
    return property;
  }
  const encoded = Buffer.from(value);
  const property = Buffer.alloc(5 + encoded.length);
  property.write('S', 0);
  property.writeUInt32LE(encoded.length, 1);
  encoded.copy(property, 5);
  return property;
}
