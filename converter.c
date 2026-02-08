#define STB_IMAGE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "stb_image.h"

// Bilinear interpolation ve grileþtirme iþlemi
void resize_and_grayscale_image(const unsigned char *input_image, int width, int height, int channels, unsigned char *output_image) {
    int new_width = 20;
    int new_height = 20;
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;
    int i, j, c;

    for (i = 0; i < new_height; i++) {
        for (j = 0; j < new_width; j++) {
            int nearest_x = (int)(j * x_ratio);
            int nearest_y = (int)(i * y_ratio);
            int src_index = (nearest_y * width + nearest_x) * channels;
            int dest_index = i * new_width + j;

            if (channels == 3 || channels == 4) { // Renkli görüntü (RGB veya RGBA)
                unsigned char r = input_image[src_index];
                unsigned char g = input_image[src_index + 1];
                unsigned char b = input_image[src_index + 2];
                unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
                output_image[dest_index] = gray;
            } else { // Zaten gri tonlamalý görüntü
                output_image[dest_index] = input_image[src_index];
            }
        }
    }
}

void write_image(const char *output_path, unsigned char *image, int width, int height) {
    FILE *file = fopen(output_path, "wb");
    if (!file) {
        printf("Görüntü kaydedilemedi: %s\n", output_path);
        return;
    }

    // Basit bir PGM (Portable Gray Map) formatýnda kaydetme
    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(unsigned char), width * height, file);
    fclose(file);
}

void process_images_in_directory(const char *input_directory, const char *output_directory) {
    struct dirent *entry;
    DIR *dp = opendir(input_directory);
    int width, height, channels, nearest_x, nearest_y, src_index, dest_index;

    if (dp == NULL) {
        printf("Dizin açýlamadý: %s\n", input_directory);
        return;
    }

    while ((entry = readdir(dp))) {
        if (entry->d_name[0] == '.') {
            continue; // Gizli dosyalarý veya geçersiz giriþleri atla
        }

        char input_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_directory, entry->d_name);

        unsigned char *input_image = stbi_load(input_path, &width, &height, &channels, 0);
        if (input_image == NULL) {
            printf("Görüntü yüklenemedi: %s\n", input_path);
            continue;
        }

        unsigned char output_image[20 * 20];
        resize_and_grayscale_image(input_image, width, height, channels, output_image);

        char output_path[512];
        snprintf(output_path, sizeof(output_path), "%s/%s.pgm", output_directory, entry->d_name);

        write_image(output_path, output_image, 20, 20);

        stbi_image_free(input_image);
    }

    closedir(dp);
}

int main() {
    const char *input_directory = "C:\\Users\\faruk\\Desktop\\trousers";
    const char *output_directory = "C:\\Users\\faruk\\Desktop\\tro";

    process_images_in_directory(input_directory, output_directory);

    printf("Tüm görüntüler baþarýyla yeniden boyutlandýrýldý, grileþtirildi ve kaydedildi.\n");

    return 0;
}

