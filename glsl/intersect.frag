/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* intersect.frag: GLSL fragment shader for depth peeling
*/

#define CUDARC_MAX_MESH_TEXTURE_WIDTH 1.0

varying vec2 tid;
varying vec3 vertexPosition;
uniform vec3 eyePos;
uniform vec2 winSize;
uniform float depthPeelPass;
uniform sampler2D depthSampler;

void main(void)
{
	vec3 dir = vertexPosition - eyePos;

  //Depth peeling
  if(depthPeelPass > 0.0){
	  vec2 coord = gl_FragCoord.xy / winSize;
	  float depth = texture2D(depthSampler, coord).r;

	  if(gl_FragCoord.z <= depth)
		  discard;
  }
  
  gl_FragColor = vec4(dir, CUDARC_MAX_MESH_TEXTURE_WIDTH * tid.x + tid.y);
  //gl_FragColor = vec4(tid/100000, tid/100000, tid/100000, 1);
  //gl_FragColor = vec4(1, 0, 0, 1);
}