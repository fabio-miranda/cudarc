/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* intersect.frag: GLSL vertex shader for depth peeling
*/

varying vec2 tid;
varying vec3 vertexPosition;

void main()
{
  gl_Position =  gl_ModelViewProjectionMatrix * gl_Vertex;
  vertexPosition = gl_Vertex.xyz;
  tid = gl_MultiTexCoord0.xy;
}
