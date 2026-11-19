uniform float hairline;
#include <color_fragment>
#ifdef USE_COLOR_ALPHA
	float tipPx = max( fwidth( vAlphaMapUv.x ) * #, fwidth( vColor.a ) * # );
	float tipFar = smoothstep( #, #, tipPx );
	float tipGain = # / mix( #, #, tipFar );
	float tipIn = ( # - smoothstep( #, #, vColor.a ) ) * ( # - # * tipFar );
	vec4 tipS = texture2D( alphaMap, vec2( vAlphaMapUv.x, vColor.a ) );
	vec4 tipS2 = texture2D( alphaMap, vec2( vAlphaMapUv.x + #, max( vColor.a - #, # ) ) );
	vec4 tipS3 = texture2D( alphaMap, vec2( vAlphaMapUv.x + #, max( vColor.a - #, # ) ) );
	diffuseColor.rgb *= #;
	float tipA = tipS.g * tipGain, tipA2 = tipS2.b * tipGain, tipA3 = tipS3.b * tipGain;
	if ( hairline > # ) { tipA = texture2D( alphaMap, vec2( vAlphaMapUv.x, vColor.a ), # ).b * #; tipA2 = #; tipA3 = #; }
	if ( tipA >= # ) {
		diffuseColor.a = tipA; diffuseColor.rgb *= mix( #, tipS.r * #, tipIn );
	} else if ( tipA2 >= # ) {
		diffuseColor.a = tipA2; diffuseColor.rgb *= #;
	} else {
		diffuseColor.a = tipA3; diffuseColor.rgb *= #;
	}
#endif
