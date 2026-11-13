// GLSL building blocks shared by every custom ShaderMaterial in the world.
export const GLSL_NOISE = `
vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
float n_hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float n_hash13(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
vec2 n_hash22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
float n_noise(vec2 p) {
vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
return mix(mix(n_hash12(i), n_hash12(i + vec2(1.0, 0.0)), u.x), mix(n_hash12(i + vec2(0.0, 1.0)), n_hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}
float n_noise3(vec3 p) {
vec3 i = floor(p), f = fract(p); vec3 u = f * f * (3.0 - 2.0 * f);
float a = mix(n_hash13(i), n_hash13(i + vec3(1.0, 0.0, 0.0)), u.x);
float b = mix(n_hash13(i + vec3(0.0, 1.0, 0.0)), n_hash13(i + vec3(1.0, 1.0, 0.0)), u.x);
float c = mix(n_hash13(i + vec3(0.0, 0.0, 1.0)), n_hash13(i + vec3(1.0, 0.0, 1.0)), u.x);
float d = mix(n_hash13(i + vec3(0.0, 1.0, 1.0)), n_hash13(i + vec3(1.0, 1.0, 1.0)), u.x);
return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}
float n_fbm(vec2 p) {
float s = 0.0, a = 0.5;
for (int i = 0; i < 4; i++) { s += a * n_noise(p); p = mat2(1.6, 1.2, -1.2, 1.6) * p; a *= 0.5; }
return s / 0.9375;
}
float n_fbm3(vec3 p) {
float s = 0.0, a = 0.5;
for (int i = 0; i < 3; i++) { s += a * n_noise3(p); p = p * 2.03 + 17.1; a *= 0.5; }
return s / 0.875;
}
// x: distance to the nearest cell point, y: distance to the edge between cells
vec2 n_cells(vec2 p) {
vec2 i = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0;
for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
  vec2 g = vec2(float(x), float(y));
  float d = length(g + n_hash22(i + g) - f);
  if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
}
return vec2(d1, d2 - d1);
}
// same, plus z: a random number that is constant across each cell
vec3 n_cells3(vec2 p) {
vec2 i = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0, id = 0.0;
for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
  vec2 g = vec2(float(x), float(y));
  vec2 o = n_hash22(i + g);
  float d = length(g + o - f);
  if (d < d1) { d2 = d1; d1 = d; id = n_hash12(i + g + 0.37); } else if (d < d2) { d2 = d; }
}
return vec3(d1, d2 - d1, id);
}
`;

export const GLSL_SKY = `
uniform vec3 uSkyTop;
uniform vec3 uHorizon;
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform float uNight;
uniform float uTime;
uniform float uExposure;
vec3 skyColour(vec3 d) {
vec3 top = lin(uSkyTop), hor = lin(uHorizon);
vec3 c = mix(hor, top, smoothstep(0.0, 0.55, d.y));
c *= mix(0.8, 1.0, smoothstep(-0.3, 0.0, d.y));
float s = max(dot(d, uSunDir), 0.0);
c += uSunCol * (pow(s, 5.0) * 0.12 + pow(s, 48.0) * 0.3) * (1.0 - uNight * 0.6);
return c;
}
vec3 applyFog(vec3 c) {
#ifdef USE_FOG
return mix(c, lin(fogColor), smoothstep(fogNear, fogFar, vFogDepth));
#else
return c;
#endif
}
float fogAmount() {
#ifdef USE_FOG
return smoothstep(fogNear, fogFar, vFogDepth);
#else
return 0.0;
#endif
}
// the world's exposure and filmic tone curve (ACES), in linear light
vec3 toneMap(vec3 c) {
c *= uExposure;
return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
}
vec4 outCol(vec3 c, float a) {
return linearToOutputTexel(vec4(toneMap(c), a));
}
`;

export const V_LIT = `
#include <common>
#include <shadowmap_pars_vertex>
uniform float uTime;
varying vec3 vWorldPos;
varying vec3 vNormalW;
varying vec3 vViewPos;
varying float vFogDepth;
mat4 modelOf() {
#ifdef USE_INSTANCING
return modelMatrix * instanceMatrix;
#else
return modelMatrix;
#endif
}
// sideways wind offset for a point hgt above the root of a tree standing at base
vec2 windSway(vec3 base, float hgt) {
float ph = base.x * 0.21 + base.z * 0.17;
float sway = sin(uTime * 1.2 + ph) * 0.6 + sin(uTime * 2.9 + ph * 1.7) * 0.25;
return vec2(0.8, 0.5) * sway * hgt * hgt * 0.004;
}
`;
export const V_END = `
vWorldPos = worldPosition.xyz;
vNormalW = worldNormal;
vec4 mvPosition = viewMatrix * worldPosition;
vViewPos = -mvPosition.xyz;
vFogDepth = -mvPosition.z;
gl_Position = projectionMatrix * mvPosition;
vec3 transformedNormal = normalize((viewMatrix * vec4(worldNormal, 0.0)).xyz);
#include <shadowmap_vertex>
`;

export const F_LIT = `
#include <common>
#include <packing>
#include <bsdfs>
#include <lights_pars_begin>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
uniform vec3 fogColor;
uniform float fogNear;
uniform float fogFar;
varying vec3 vWorldPos;
varying vec3 vNormalW;
varying vec3 vViewPos;
varying float vFogDepth;
` + GLSL_NOISE + GLSL_SKY + `
// Diffuse light arriving at a surface: hemisphere sky, the sun with its shadow, and the fires.
// wrap > 0 lets light bend past the terminator, for foliage, cloth and skin.
vec3 gather(vec3 Nw, float wrap, out float sh) {
vec3 N = normalize((viewMatrix * vec4(Nw, 0.0)).xyz);
vec3 vp = -vViewPos;
vec3 irr = vec3(0.0);
sh = 1.0;
#ifdef USE_SHADOWMAP
sh = getShadowMask();
#endif
#if NUM_HEMI_LIGHTS > 0
for (int i = 0; i < NUM_HEMI_LIGHTS; i++) {
  float w = dot(N, hemisphereLights[i].direction) * 0.5 + 0.5;
  irr += mix(hemisphereLights[i].groundColor, hemisphereLights[i].skyColor, w);
}
#endif
#if NUM_DIR_LIGHTS > 0
for (int i = 0; i < NUM_DIR_LIGHTS; i++) {
  float nl = clamp((dot(N, directionalLights[i].direction) + wrap) / (1.0 + wrap), 0.0, 1.0);
  irr += directionalLights[i].color * nl * sh;
}
#endif
#if NUM_POINT_LIGHTS > 0
for (int i = 0; i < NUM_POINT_LIGHTS; i++) {
  vec3 L = pointLights[i].position - vp;
  float d = length(L);
  float att = pointLights[i].distance > 0.0 ? pow(clamp(1.0 - d / pointLights[i].distance, 0.0, 1.0), pointLights[i].decay) : 1.0;
  float nl = clamp((dot(N, L / d) + wrap) / (1.0 + wrap), 0.0, 1.0);
  irr += pointLights[i].color * att * nl;
}
#endif
return irr;
}
`;

export const V_UNLIT = `
#include <common>
uniform float uTime;
uniform float uViewH;
varying float vFogDepth;
`;
export const F_UNLIT = `
#include <common>
uniform vec3 fogColor;
uniform float fogNear;
uniform float fogFar;
varying float vFogDepth;
` + GLSL_NOISE + GLSL_SKY;

// Vertex shader shared by objects that texture themselves in their own local space.
export const V_LOCAL = `
varying vec3 vLocal;
varying vec3 vLocalN;
void main() {
mat4 m = modelOf();
vec4 worldPosition = m * vec4(position, 1.0);
vec3 worldNormal = normalize(mat3(m) * normal);
vLocal = position;
vLocalN = normal;
` + V_END + `}`;
