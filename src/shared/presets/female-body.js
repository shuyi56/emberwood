// Female figure preset: the slender, long-limbed build first made for the Ranger.
// Spread it into a character spec ({ ...FEMALE_BODY, ... }) and override any field after it.
// `body` holds the proportions drawFigure reads (half-widths and heights in sprite pixels):
//   curve     control points [height, half-width] for the torso outline, hip -> waist -> bust
//   belt      height of the belt line
//   neckDrop  how far the neckline drops below the default
//   sh, chest, bust, waist, hip   half-widths at those heights
//   depth     front-to-back depth of the torso
//   arm, fore leg, shin           limb widths (upper arm, forearm, thigh, shin)
//   gap       how far apart the legs are set
//   slope     how steeply the shoulders fall
//   armOut    how far the arms hang out from the body
const FEMALE_BODY = {
  // narrow waist, fuller hips and bust, slimmer arms and legs, legs set closer
  body: { curve: [[15, 8], [20.5, 9.1], [27.6, 3.9], [34, 8.4]], belt: 26, neckDrop: 0.8, sh: 7.8, chest: 7.2, bust: 8.6, waist: 4.1, hip: 9.2, depth: 5, arm: 4, fore: 3.4, leg: 5.8, shin: 4.2, gap: 3.6, slope: 3.4, armOut: 0.2 },
  // arms with soft sleeve caps rather than hard contoured edges
  softArms: true
};

export { FEMALE_BODY };
