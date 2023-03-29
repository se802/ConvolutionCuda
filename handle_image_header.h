//
// Created by stelios on 25/01/2023.
//

#ifndef HOMEWORK3_HANDLE_IMAGE_HEADER_H
#define HOMEWORK3_HANDLE_IMAGE_HEADER_H

#include "image_types.h"

int load_bmp_headers(BMP_FILE_HEADER *BFH, BMP_INFO_HEADER *BIH, FILE *fp);
void print_header_info(BMP_FILE_HEADER *fileHeader, BMP_INFO_HEADER *infoHeader);
int writeBmpInformation(BMP_FILE_HEADER *fileHeader, BMP_INFO_HEADER *infoHeader, FILE *file);


// Check that the file is a BMP file.
bool fileNotBMP(BMP_FILE_HEADER *BFH){
    if (BFH->fileType1 != 'B' || BFH->fileType2 != 'M')
        return true;
    return false;
}

void readBMPinformationHeader(BMP_FILE_HEADER *BFH, BMP_INFO_HEADER *BIH, FILE *fp){
    // Load the BMP information header.
    fread((void *) &(BFH->fileSize), sizeof(uint), 1, fp);
    fread((void *) &(BFH->reserved1), sizeof(ushort), 1, fp);
    fread((void *) &(BFH->reserved2), sizeof(ushort), 1, fp);
    fread((void *) &(BFH->imageDataOffset), sizeof(uint), 1, fp);

    fread((void *) &(BIH->headerSize), sizeof(uint), 1, fp);
    fread((void *) &(BIH->imageWidth), sizeof(uint), 1, fp);
    fread((void *) &(BIH->imageHeight), sizeof(uint), 1, fp);
    fread((void *) &(BIH->colorPlanes), sizeof(ushort), 1, fp);
    fread((void *) &(BIH->bitsPerPixel), sizeof(ushort), 1, fp);
}

void readColorPixelInfo(BMP_FILE_HEADER *BFH, BMP_INFO_HEADER *BIH, FILE *fp){
    fread((void *) &(BIH->imageSize), sizeof(uint), 1, fp);
    fread((void *) &(BIH->pixelsPerMeterWidth), sizeof(uint), 1, fp);
    fread((void *) &(BIH->pixelsPerMeterHeight), sizeof(uint), 1, fp);
    fread((void *) &(BIH->numColorsInPalette), sizeof(uint), 1, fp);
    fread((void *) &(BIH->numImportantColors), sizeof(uint), 1, fp);
}

// Loads the BMP file header and information header from the given file.
// Returns 0 if the file is loaded successfully, 1 otherwise.
int load_bmp_headers(BMP_FILE_HEADER *BFH, BMP_INFO_HEADER *BIH, FILE *fp) {
    // Load the BMP file header.
    fread((void *) &(BFH->fileType1), sizeof(myByte), 1, fp);
    fread((void *) &(BFH->fileType2), sizeof(myByte), 1, fp);

    if (fileNotBMP(BFH))
        return 1;

    // Load the BMP information header.
    readBMPinformationHeader(BFH,BIH,fp);

    // Check that the image is 24-bit.
    if (BIH->bitsPerPixel != 24)
        return 1;

    fread((void *) &(BIH->compressionType), sizeof(uint), 1, fp);

    // Check that the image is not compressed.
    if (BIH->compressionType != 0)
        return (1);

    readColorPixelInfo(BFH,BIH,fp);

    // Return 0 to indicate success.
    return 0;
}



// Prints the BMP file header to the console.
void print_file_header_info(BMP_FILE_HEADER* header) {
    printf("\nBMP File Header\n");
    printf("================\n");
    printf("File Type: %c%c\n", header->fileType1, header->fileType2);
    printf("File Size: %d\n", header->fileSize);
    printf("Reserved 1: %d\n", header->reserved1);
    printf("Reserved 2: %d\n", header->reserved2);
    printf("Image Data Offset: %d\n", header->imageDataOffset);
}

// Prints the BMP information header to the console.
void print_info_header_info(BMP_INFO_HEADER* header) {
    printf("\nBMP Information Header\n");
    printf("=======================\n");
    printf("Header Size: %d\n", header->headerSize);
    printf("Image Width: %d\n", header->imageWidth);
    printf("Image Height: %d\n", header->imageHeight);
    printf("Color Planes: %d\n", header->colorPlanes);
    printf("Bits Per Pixel: %d\n", header->bitsPerPixel);
    printf("Compression Type: %d\n", header->compressionType);
    printf("Image Size: %d\n", header->imageSize);
    printf("Pixels Per Meter Width: %d\n", header->pixelsPerMeterWidth);
    printf("Pixels Per Meter Height: %d\n", header->pixelsPerMeterHeight);
    printf("Number of Colors in Palette: %d\n", header->numColorsInPalette);
    printf("Number of Important Colors: %d\n", header->numImportantColors);
    printf("\n***************************************************************************\n");
}


// Prints the BMP file header and information header to the console.
void print_header_info(BMP_FILE_HEADER* fileHeader, BMP_INFO_HEADER* infoHeader) {
    print_file_header_info(fileHeader);
    print_info_header_info(infoHeader);
}




// Writes the BMP file header to the given file.
void writeBMPfileHeader(BMP_FILE_HEADER *fileHeader, FILE *file) {
    fwrite((void *) &(fileHeader->fileType1), sizeof(myByte), 1, file);
    fwrite((void *) &(fileHeader->fileType2), sizeof(myByte), 1, file);
    fwrite((void *) &(fileHeader->fileSize), sizeof(uint), 1, file);
    fwrite((void *) &(fileHeader->reserved1), sizeof(ushort), 1, file);
    fwrite((void *) &(fileHeader->reserved2), sizeof(ushort), 1, file);
    fwrite((void *) &(fileHeader->imageDataOffset), sizeof(uint), 1, file);
}

// Writes the BMP information header to the given file.
void writeBMPinformationHeader(BMP_INFO_HEADER *infoHeader, FILE *file) {
    fwrite((void *) &(infoHeader->headerSize), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->imageWidth), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->imageHeight), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->colorPlanes), sizeof(ushort), 1, file);
    fwrite((void *) &(infoHeader->bitsPerPixel), sizeof(ushort), 1, file);
    fwrite((void *) &(infoHeader->compressionType), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->imageSize), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->pixelsPerMeterWidth), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->pixelsPerMeterHeight), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->numColorsInPalette), sizeof(uint), 1, file);
    fwrite((void *) &(infoHeader->numImportantColors), sizeof(uint), 1, file);
}


// Writes a new BMP file header and information header to the given file.
// Returns 0 if the headers are written successfully, 1 otherwise.
int writeBmpInformation(BMP_FILE_HEADER *fileHeader, BMP_INFO_HEADER *infoHeader, FILE *file) {
    // Write the BMP file header.
    writeBMPfileHeader(fileHeader, file);

    // Write the BMP information header.
    writeBMPinformationHeader(infoHeader, file);

    // Return 0 to indicate success.
    return 0;
}

#endif //HOMEWORK3_HANDLE_IMAGE_HEADER_H
