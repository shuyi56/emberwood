// A small glTF 2.0 binary (.glb) writer and reader for the hair asset (FORMAT.md). No dependencies: the asset is
// plain glTF, so any viewer opens it, and the Emberwood readers (hair-asset.js, blender/shared/hair_asset.py) read
// the hair's own data from its extras and its underscore attributes.
'use strict';
const COMP = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const NCOMP = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const TYPE = { 1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4' };

class GlbWriter {
  constructor() { this.json = { asset: { version: '2.0', generator: 'emberwood hair-asset' }, buffers: [{ byteLength: 0 }],
    bufferViews: [], accessors: [] }; this.chunks = []; this.len = 0; }
  view(bytes, target) {
    const pad = (4 - (this.len % 4)) % 4;
    if (pad) { this.chunks.push(Buffer.alloc(pad)); this.len += pad; }
    const v = { buffer: 0, byteOffset: this.len, byteLength: bytes.byteLength };
    if (target) v.target = target;
    this.chunks.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength)); this.len += bytes.byteLength;
    this.json.bufferViews.push(v); return this.json.bufferViews.length - 1;
  }
  // a typed array as an accessor: n components per element
  accessor(arr, n, opts) {
    opts = opts || {};
    const ct = arr instanceof Float32Array ? 5126 : arr instanceof Uint32Array ? 5125 : arr instanceof Uint16Array ? 5123 : 5121;      // (float, uint32, uint16, uint8)
    const a = { bufferView: this.view(arr, opts.target), componentType: ct, count: arr.length / n, type: TYPE[n] };
    if (opts.minmax) {
      const mn = new Array(n).fill(Infinity), mx = new Array(n).fill(-Infinity);
      for (let i = 0; i < arr.length; i++) { const k = i % n; if (arr[i] < mn[k]) mn[k] = arr[i]; if (arr[i] > mx[k]) mx[k] = arr[i]; }
      a.min = mn; a.max = mx;
    }
    if (opts.normalized) a.normalized = true;
    this.json.accessors.push(a); return this.json.accessors.length - 1;
  }
  image(png, name) {
    this.json.images = this.json.images || [];
    this.json.images.push({ bufferView: this.view(png), mimeType: 'image/png', name: name });
    return this.json.images.length - 1;
  }
  write() {
    this.json.buffers[0].byteLength = this.len;
    let js = Buffer.from(JSON.stringify(this.json), 'utf8');
    const jp = (4 - (js.length % 4)) % 4; if (jp) js = Buffer.concat([js, Buffer.alloc(jp, 0x20)]);
    let bin = Buffer.concat(this.chunks); const bp = (4 - (bin.length % 4)) % 4; if (bp) bin = Buffer.concat([bin, Buffer.alloc(bp)]);
    const h = Buffer.alloc(12); h.writeUInt32LE(0x46546C67, 0); h.writeUInt32LE(2, 4); h.writeUInt32LE(12 + 8 + js.length + 8 + bin.length, 8);
    const c0 = Buffer.alloc(8); c0.writeUInt32LE(js.length, 0); c0.writeUInt32LE(0x4E4F534A, 4);
    const c1 = Buffer.alloc(8); c1.writeUInt32LE(bin.length, 0); c1.writeUInt32LE(0x004E4942, 4);
    return Buffer.concat([h, c0, js, c1, bin]);
  }
}

// read a .glb: { json, bin, accessor(i) -> typed array (tightly packed), image(i) -> Buffer }
function readGlb(buf) {
  if (buf.readUInt32LE(0) !== 0x46546C67) throw new Error('not a glb');
  let off = 12, json = null, bin = null;
  while (off < buf.length) {
    const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4), body = buf.subarray(off + 8, off + 8 + len);
    if (type === 0x4E4F534A) json = JSON.parse(body.toString('utf8')); else if (type === 0x004E4942) bin = body;
    off += 8 + len;
  }
  return {
    json, bin,
    accessor(i) {
      const a = json.accessors[i], v = json.bufferViews[a.bufferView], T = COMP[a.componentType], n = NCOMP[a.type];
      const bytes = bin.subarray(v.byteOffset + (a.byteOffset || 0), v.byteOffset + (a.byteOffset || 0) + a.count * n * T.BYTES_PER_ELEMENT);
      const out = new T(a.count * n); Buffer.from(out.buffer).set(bytes); return out;
    },
    image(i) { const v = json.bufferViews[json.images[i].bufferView]; return bin.subarray(v.byteOffset, v.byteOffset + v.byteLength); },
  };
}
module.exports = { GlbWriter, readGlb };
