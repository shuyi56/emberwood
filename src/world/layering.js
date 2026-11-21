// Layering: a part of the body drawn a layer under the clothes. Where an arm swings into a cloak or a cape that
// rides the chest, or a knee lifts into its skirt, the skin pokes a little way through the cloth; here the skin there
// is given a depth a few units further from the camera than it really is (its place on screen is unchanged), so
// the cloth over it always wins the depth test. Only depth moves: what lies clear of the cloth draws as before, and
// a hand or a foot out in front of the cloak is still in front of it.
//
//   underClothes(bodyMesh)                                   // the arms and the legs (LIMBS)
//   underClothes(bodyMesh, { bones: /^Thigh/, depth: 1 })    // another part, another depth
//
// bones: the joints whose skin goes under (matched against the skeleton's bone names, which GLTFLoader gives without
// their dots: `UpperArm.L` arrives as `UpperArmL`); each vertex goes under by its skin weight on them, so the layer
// fades out over the joints it shares with the rest of the body. depth: how far under, in the model's own units
// (scaled with the model in the world, as the cape solver's units are).
import * as THREE from 'three';

export const LIMBS = /^(UpperArm|Forearm|Thigh|Shin)/;

export function underClothes(mesh, { bones = LIMBS, depth = 3 } = {}) {
  if (!mesh.isSkinnedMesh || !mesh.skeleton) return false;
  const flags = mesh.skeleton.bones.map((b) => (bones.test(b.name) ? 1 : 0));
  if (!flags.some((f) => f)) return false;
  mesh.updateMatrixWorld(true);
  const scale = new THREE.Vector3().setFromMatrixScale(mesh.parent ? mesh.parent.matrixWorld : mesh.matrixWorld).x || 1;
  const n = flags.length;
  const uniforms = { layerBone: { value: flags }, layerDepth: { value: depth * scale } };
  const mats = [].concat(mesh.material).map((m) => {
    const x = m.clone();
    x.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\nuniform float layerBone[${n}];\nuniform float layerDepth;`)
        .replace('#include <project_vertex>', `#include <project_vertex>
  #ifdef USE_SKINNING
  {
    // (the share of this vertex that belongs to the layered joints, by its skin weights)
    float under = dot(skinWeight, vec4(layerBone[int(skinIndex.x)], layerBone[int(skinIndex.y)],
                                       layerBone[int(skinIndex.z)], layerBone[int(skinIndex.w)]));
    if (under > 0.0) {
      // the same point moved back along its line of sight: its depth from there, its place on screen kept
      float far = length(mvPosition.xyz);
      vec4 back = projectionMatrix * vec4(mvPosition.xyz * (1.0 + under * layerDepth / max(far, 1e-3)), 1.0);
      gl_Position.z = back.z / back.w * gl_Position.w;
    }
  }
  #endif`);
    };
    // (its own program: the layered shader must not be shared with an unlayered copy of the same material)
    x.customProgramCacheKey = () => `under-${n}-${bones}`;
    return x;
  });
  mesh.material = Array.isArray(mesh.material) ? mats : mats[0];
  mesh.userData.layer = { bones: String(bones), depth };
  return true;
}
