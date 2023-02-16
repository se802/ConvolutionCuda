//
// Created by stelios on 25/01/2023.
//

#ifndef HOMEWORK3_HANDLE_IMAGE_HEADER_H
#define HOMEWORK3_HANDLE_IMAGE_HEADER_H

#include "image_types.h"

int load_HEADER(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH, FILE *fp);
void LIST(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH);
int NEW_FILE_HEADER(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH,FILE *new_fp);
//SINARTISI POY DIABAZI TO HEADER ENOS ARXEIOY 'BMP'.
//DEXETE OS ORIZMATA TA DIO MERI TOY HEADER KAI TO ARXEIO.
//EPISTREFI 0 GIA OMALI EKTELESI KAI 1 GIA PROBLIMATIKI.
int load_HEADER(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH, FILE *fp) {
    //BITMAP FILE HEADER
    fread((void *) &(BFH->bfType1), sizeof(byte), 1, fp);
    fread((void *) &(BFH->bfType2), sizeof(byte), 1, fp);

    //EAN TO ARXIO DEN EINAI 'BMP' EPISTREFI 1-PROBLIMA.
    if (BFH->bfType1 != 'B' || BFH->bfType2 != 'M')
        return (1);

    fread((void *) &(BFH->bfSize), sizeof(dword), 1, fp);
    fread((void *) &(BFH->bfReserved1), sizeof(word), 1, fp);
    fread((void *) &(BFH->bfReserved2), sizeof(word), 1, fp);
    fread((void *) &(BFH->dfOffBits), sizeof(dword), 1, fp);
    //BITMAP INFO HEADER.
    fread((void *) &(BIH->biSize), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biWidth), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biHeight), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biPlanes), sizeof(word), 1, fp);
    fread((void *) &(BIH->biBitCount), sizeof(word), 1, fp);

    //EAN DEN EINAI ARXIO 24-bit EPISTREFI 1-PROBLIMA.
    if (BIH->biBitCount != 24)
        return (1);

    fread((void *) &(BIH->biCompression), sizeof(dword), 1, fp);

    //EAN EINAI SIMPIEZMENO ARXIO EPISTREFI 1-PROBLIMA.
    if (BIH->biCompression != 0)
        return (1);

    fread((void *) &(BIH->biSizeImage), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biXPelsPerMeter), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biYPelsPerMeter), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biClrUsed), sizeof(dword), 1, fp);
    fread((void *) &(BIH->biClrImportant), sizeof(dword), 1, fp);

    //OTAN GINI KANONIKA H ANAGNOSI TOY ARXIOY EPISTREFI 0-KANONIKA.
    return (0);
}


//DEXETE OS ORIZMATA TA DIO MERI TOY HEADER KAI TA EKTIPONI.
void LIST(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH) {
    printf("\nBITMAP_FILE_HEADER\n");
    printf("==================\n");
    printf("bfType: %c%c\n", BFH->bfType1, BFH->bfType2);
    printf("bfSize: %d\n", BFH->bfSize);
    printf("bfReserved1: %d\n", BFH->bfReserved1);
    printf("bfReserved2: %d\n", BFH->bfReserved2);
    printf("bfOffBits: %d\n", BFH->dfOffBits);
    printf("\nBITMAP_INFO_HEADER\n");
    printf("==================\n");
    printf("biSize: %d\n", BIH->biSize);
    printf("biWidth: %d\n", BIH->biWidth);
    printf("biHeight: %d\n", BIH->biHeight);
    printf("biPlanes: %d\n", BIH->biPlanes);
    printf("biBitCount: %d\n", BIH->biBitCount);
    printf("biCompression: %d\n", BIH->biCompression);
    printf("biSizeImage: %d\n", BIH->biSizeImage);
    printf("biXPelsPerMeter: %d\n", BIH->biXPelsPerMeter);
    printf("biYPelsPerMeter: %d\n", BIH->biYPelsPerMeter);
    printf("biClrUsed: %d\n", BIH->biClrUsed);
    printf("biClrImportant: %d\n", BIH->biClrImportant);
    printf("\n***************************************************************************\n");
}


//SINARTISI POY DIMIOYRGA TO NEO FILE.
int NEW_FILE_HEADER(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH,FILE *new_fp) {

    fwrite((void *) &(BFH->bfType1), sizeof(byte), 1, new_fp);
    fwrite((void *) &(BFH->bfType2), sizeof(byte), 1, new_fp);
    fwrite((void *) &(BFH->bfSize), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BFH->bfReserved1), sizeof(word), 1, new_fp);
    fwrite((void *) &(BFH->bfReserved2), sizeof(word), 1, new_fp);
    fwrite((void *) &(BFH->dfOffBits), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biSize), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biWidth), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biHeight), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biPlanes), sizeof(word), 1, new_fp);
    fwrite((void *) &(BIH->biBitCount), sizeof(word), 1, new_fp);
    fwrite((void *) &(BIH->biCompression), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biSizeImage), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biXPelsPerMeter), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biYPelsPerMeter), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biClrUsed), sizeof(dword), 1, new_fp);
    fwrite((void *) &(BIH->biClrImportant), sizeof(dword), 1, new_fp);
    return (0);
}


#endif //HOMEWORK3_HANDLE_IMAGE_HEADER_H
