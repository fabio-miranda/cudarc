/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* transferfunc.cpp: load transfer function files
*/


#include "transferfunc.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

TpvColorScale* TransferFunc(const char* tffilepath, float scalarmin, float scalarmax){

  printf("Transfer function file: %s ... ", tffilepath);

  FILE* fp = fopen(tffilepath, "r");
  if (!fp) {
    fprintf(stderr, "Error: cannot read transfer function file.\n");
    exit(-1);
  }

  //Read transfer function file.
  //Skip the lines that starts with #
  int numbpoints = 0;
  char line[256];
  int interpmode = -1;

  fscanf(fp, " %s", &line);
  while(strcmp(&line[0], "#") == 0)
    fscanf(fp, " %s", &line);
  
  if(strcmp(line, "constant") == 0 || strcmp(line, "const") == 0)
    interpmode = TpvColorScale::INTERP_CONSTANT;
  else if(strcmp(line, "discrete") == 0)
    interpmode = TpvColorScale::INTERP_DISCRETE;
  else if(strcmp(line, "linear") == 0)
    interpmode = TpvColorScale::INTERP_LINEAR;
  else{
    fprintf(stderr, "Error: cannot read transfer function file.\n");
    exit(-1);
  }

  fscanf(fp, " %s", &line);
  numbpoints = atoi(line);
  
  
  TpvColorScale* colorscale = new TpvColorScale(numbpoints);
  colorscale->SetInterpolationMode(interpmode);
  colorscale->Reset(numbpoints);
  for(int i=0; i<numbpoints; i++){
    float color[4];
    fscanf(fp, "%f %f %f %f", &color[0], &color[1], &color[2], &color[3]);
    colorscale->SetColor(i, color[0], color[1], color[2], color[3]);
  }

  fclose(fp);

  printf("Done.\n");

  if(scalarmin != scalarmax)
    colorscale->SetValueLimits(scalarmin, scalarmax);

  return colorscale;
}


TpvColorScale* IsoValues(const char* isofilepath, float scalarmin, float scalarmax, float** isovalues){

  int numbpoints = 0;
  char line[256];
  int interpmode = -1;
  TpvColorScale* colorscale;

  printf("Iso values file: %s ... ", isofilepath);

  FILE* fp = fopen(isofilepath, "r");
  if (!fp) {
    fprintf(stderr, "Error: cannot read iso values file.\n");
    exit(-1);
  }

  //Read tiso values file.
  //Skip the lines that starts with #
  fscanf(fp, " %s", &line);
  while(strcmp(&line[0], "#") == 0)
    fscanf(fp, " %s", &line);

  numbpoints = atoi(line);
  colorscale = new TpvColorScale();
  *isovalues = new float[numbpoints];
  colorscale->SetInterpolationMode(TpvColorScale::INTERP_CONSTANT);
  colorscale->Reset(numbpoints);

  for(int i=0; i<numbpoints; i++){
    float scalar;
    float color[4];

    fscanf(fp, "%f %f %f %f %f", &scalar, &color[0], &color[1], &color[2], &color[3]);
    //for(int j=-2; j<2; j++){
      colorscale->SetColor(i , color[0], color[1], color[2], color[3]);
    //}
      (*isovalues)[i] = scalar;
  }

  fclose(fp);

  printf("Done.\n");

  if(scalarmin != scalarmax)
    colorscale->SetValueLimits(scalarmin, scalarmax);

  return colorscale;
}
