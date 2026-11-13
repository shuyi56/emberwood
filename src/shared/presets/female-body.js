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
//   legLen    extra leg length (see View): the figure stands taller on longer legs
const FEMALE_BODY = {
  // narrow waist, fuller hips and bust, slimmer arms and legs, legs set closer
  // (narrow shoulders to go with the slim hips, as the full sleeves stand out from them; and the hips no
  // wider than the shoulders: wider, the seat and the jerkin over it read as a broad,
  // heavy shape from behind, where the hair hides the waist, and the jerkin flares out like a skirt)
  body: { curve: [[15, 6.9], [20.5, 7.6], [27.6, 3.9], [34, 7.6]], belt: 26, neckDrop: 0.8, sh: 7.1, chest: 7.0, bust: 7.8, waist: 4.1, hip: 7.7, depth: 5, arm: 4, fore: 3.4, leg: 5.8, shin: 4.2, gap: 3.6, slope: 3.4, armOut: 0.2, legLen: 2 },
  // arms with soft sleeve caps rather than hard contoured edges
  softArms: true,
  // smaller hands, in proportion to the slim forearms (see drawFigure)
  handScale: 0.78
};

export { FEMALE_BODY };
