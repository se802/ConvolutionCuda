//
// Created by stelios on 25/01/2023.
//

#ifndef HOMEWORK3_IMAGE_TYPES_H
#define HOMEWORK3_IMAGE_TYPES_H

#include "stdio.h"

typedef unsigned char myByte;
typedef unsigned int uint;
typedef unsigned short int ushort;

//Struct that has the 3 colors of the pixel
typedef struct {
    myByte R;
    myByte G;
    myByte B;
}__attribute__((packed)) pixel;


typedef struct {
    float **kernel;
    float bias;
    float factor;
} kernel;





typedef struct {
    myByte fileType1;         // The first byte of the file type.
    myByte fileType2;         // The second byte of the file type.
    uint fileSize;           // The total file size in bytes.
    ushort reserved1;      // Reserved field (currently unused).
    ushort reserved2;      // Reserved field (currently unused).
    uint imageDataOffset;         // The offset to the beginning of the image data.
} __attribute__((packed)) BMP_FILE_HEADER;

// Struct that has the BMP_INFO_HEADER data
typedef struct {
    uint headerSize;           // The size of this header in bytes.
    uint imageWidth;           // The width of the bitmap image in pixels.
    uint imageHeight;          // The height of the bitmap image in pixels.
    ushort colorPlanes;        // The number of color planes being used (usually 1).
    ushort bitsPerPixel;       // The number of bits per pixel.
    uint compressionType;      // The type of compression being used.
    uint imageSize;            // The size of the image data in bytes.
    uint pixelsPerMeterWidth;  // The horizontal resolution of the image in pixels per meter.
    uint pixelsPerMeterHeight; // The vertical resolution of the image in pixels per meter.
    uint numColorsInPalette;   // The number of colors in the color palette.
    uint numImportantColors;   // The number of important colors used (0 means all colors are important).
} __attribute__((packed)) BMP_INFO_HEADER;

#endif //HOMEWORK3_IMAGE_TYPES_H
