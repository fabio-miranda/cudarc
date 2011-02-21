/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* zetapsigamma.cpp: load zeta psi gamma files
*/


#include "zetapsigamma.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void PsiGamma(const char* zetapsigammapath, int size, float* data){

  const char* filename;
  char path[256];

  switch(size)
  {
  case 4:
    filename = PSIGAMMA_4_FILENAME;
    break;
  case 16:
    filename = PSIGAMMA_16_FILENAME;
    break;
  case 32:
    filename = PSIGAMMA_32_FILENAME;
    break;
  case 64:
    filename = PSIGAMMA_64_FILENAME;
  	break;
  case 128:
    filename = PSIGAMMA_128_FILENAME;
  	break;
  case 256:
    filename = PSIGAMMA_256_FILENAME;
    break;
  case 512:
    filename = PSIGAMMA_512_FILENAME;
    break;
  case 1024:
    filename = PSIGAMMA_1024_FILENAME;
    break;
  default:
    filename = PSIGAMMA_512_FILENAME;
    break;
  }

  sprintf(path, "%s", zetapsigammapath);
  sprintf(path+strlen(path), filename);
  printf("PsiGamma %dx%d: %s ... ", size, size, path);

  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: cannot read PsiGamma file.\n");
    exit(-1);
  }

  //Read PsiGamma file.
  //Skip the first lines.
  for (int i = 0; i < NUM_HEADER_LINES; i++)
    fscanf(fp, " %*[^\n]\n");
  for (int i = 0; i < size*size; i++) {
    if (fscanf(fp, " %f", &data[i]) < 1) {
      fprintf(stderr, "Error: invalid PsiGamma file.\n");
      exit(-1);
    }
  }

  fclose(fp);

  printf("Done.\n");

}

void ZetaPsiGamma(const char* zetapsigammapath, int size, float* data){


  const char* filenames[4];
  char path[4][256];

  switch(size)
  {
  case 64:
    filenames[0] = ZETA_64_FILENAME;
    filenames[1] = PSIGAMMA1_64_FILENAME;
    filenames[2] = PSIGAMMA2_64_FILENAME;
    filenames[3] = PSIGAMMA3_64_FILENAME;
    break;
  case 128:
    filenames[0] = ZETA_128_FILENAME;
    filenames[1] = PSIGAMMA1_128_FILENAME;
    filenames[2] = PSIGAMMA2_128_FILENAME;
    filenames[3] = PSIGAMMA3_128_FILENAME;
    break;
  case 256:
    filenames[0] = ZETA_256_FILENAME;
    filenames[1] = PSIGAMMA1_256_FILENAME;
    filenames[2] = PSIGAMMA2_256_FILENAME;
    filenames[3] = PSIGAMMA3_256_FILENAME;
    break;
  default:
    filenames[0] = ZETA_128_FILENAME;
    filenames[1] = PSIGAMMA1_128_FILENAME;
    filenames[2] = PSIGAMMA2_128_FILENAME;
    filenames[3] = PSIGAMMA3_128_FILENAME;
    break;
  }

  for(int i=0; i<4; i++){
    sprintf(path[i], "%s", zetapsigammapath);
    sprintf(path[i]+strlen(path[i]), filenames[i]);
  }

  printf("ZetaPsiGamma %dx%dx%d: %s ... ", size, size, size, path[0]);

  
  FILE** fp = new FILE*[4];
  for(int i=0; i<4; i++){
    fp[i] = fopen(path[i], "r");
    if (!fp[i]) {
      fprintf(stderr, "Error: cannot read ZetaPsiGamma file.\n");
      exit(-1);
    }
  }


  for(int i=0; i<size*size*size; i++){
    for(int j=0; j<4; j++){

      for (int k = 0; k < NUM_HEADER_LINES; k++)
        fscanf(fp[j], " %*[^\n]\n");

      if (fscanf(fp[j], " %f", &data[4*i +j]) < 1) {
        fprintf(stderr, "Error: invalid ZetaPsiGamma file.\n");
        exit(-1);
      }
    }
  }

  printf("Done.\n");

}