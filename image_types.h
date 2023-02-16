//
// Created by stelios on 25/01/2023.
//

#ifndef HOMEWORK3_IMAGE_TYPES_H
#define HOMEWORK3_IMAGE_TYPES_H

#include "stdio.h"

typedef unsigned char byte;
typedef unsigned int dword;
typedef unsigned short int word;

//DOMI DEDOMENO POY PERIEXEI TA TRIA XROMATA TON PIXEL.
typedef struct {
    byte R;
    byte G;
    byte B;
}__attribute__((packed)) tbyte;


typedef struct {
    float **kernel;
    float bias;
    float factor;
} kernel;

//DOMI DEDOMENO POY PERIEXEI TA STOIXIA TOY BITMAP FILE HEADER.
typedef struct {
    byte bfType1;
    byte bfType2;
    dword bfSize;
    word bfReserved1;
    word bfReserved2;
    dword dfOffBits;
}__attribute__((packed)) BITMAP_FILE_HEADER;

//DOMI DEDOMENO POY PERIEXEI TA STOIXIA TOY BITMAP INFO HEADER.
typedef struct {
    dword biSize;
    dword biWidth;
    dword biHeight;
    word biPlanes;
    word biBitCount;
    dword biCompression;
    dword biSizeImage;
    dword biXPelsPerMeter;
    dword biYPelsPerMeter;
    dword biClrUsed;
    dword biClrImportant;
}__attribute__((packed)) BITMAP_INFO_HEADER;

#endif //HOMEWORK3_IMAGE_TYPES_H
