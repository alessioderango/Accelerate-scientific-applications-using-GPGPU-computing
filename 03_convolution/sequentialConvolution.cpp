//#include <cuda.h>
#include <iostream>
#include <libpng/png.h>
#include <chrono>
using namespace std;

png_uint_32 width;
png_uint_32 height;
int color_type;
int channels ;

const int convDim = 3;

void read_png(char *file_name, png_infop& info_ptr, png_bytepp& rows_ptr)
{
    FILE *fp = fopen(file_name, "r");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);  
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    rows_ptr = png_get_rows(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height , NULL, &color_type, NULL, NULL, NULL);
    

    switch(color_type){
        case PNG_COLOR_TYPE_RGB:
            channels = 3;
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA:  
            channels = 4;
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            channels=2;
            break;
        default:
            channels = 1;
    }
    
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    fclose(fp);
}


void create_png(char *file_name, png_infop& info_ptr, png_bytepp& rows_ptr)
{
    FILE *fp = fopen(file_name, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, rows_ptr);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


void transform_image(unsigned int ** dev_recv, unsigned int** row_pointers , int height , int width, int channels, int convDim,int* conv_matrix){
    for(int i = 0; i<height;++i){   
        for(int j = 0;j<width;++j){
            int sum = 0;
            for(int k = 0;k<convDim;++k){

                int rel_y = i+k-convDim/2;
                for(int x = 0;x<convDim;++x){

                    int rel_x = j+(x-convDim/2)*(channels);

                    if(rel_y > 0 && rel_y< height && rel_x <width && rel_x > 0 )
                        sum+=row_pointers[rel_y][rel_x]*conv_matrix[k*convDim+x];
                    }
            }
            if(sum<0) sum = 0;
            if(sum>255){ sum = 255;}
            dev_recv[i][j] = sum;
        }
    }
}


int main(int argc, char *argv[]){

    //int* conv_matrix= new int[convDim*convDim]{0,-1,0,-1,5,-1, 0,-1,0};
    int* conv_matrix= new int[convDim*convDim]{-1, -1, -1, -1, 8, -1, -1, -1, -1};
 
    if(argc!=3){
        cout << "ERROR\n";
        return 0;
    }
    char* path = argv[1];
    char* path_to_write = argv[2];
    png_infop info_ptr;
    png_bytepp rows;
    read_png(path, info_ptr, rows);
      
    int rgb_width= width *channels; // 4800 - 1600*3
    unsigned linearDim = height*rgb_width; // 

    cout << " rgb_width: " << rgb_width << " linearDim: " << linearDim  << endl;
 
    unsigned int** row_pointers = new unsigned int* [height];
    unsigned int** receive_buffer = new unsigned int* [height];
    for (unsigned int y = 0; y < height; y++) {
        row_pointers[y]  = new unsigned[rgb_width];
        receive_buffer[y]= new unsigned[rgb_width];
        for (unsigned int x = 0; x < rgb_width; x++) {
            row_pointers[y][x] = rows[y][x];
        }
    }
    std::chrono::time_point<std::chrono::system_clock> start, end;
  
    start = std::chrono::system_clock::now();
  
    transform_image(receive_buffer , row_pointers, height, rgb_width, channels, convDim,conv_matrix);
        end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < rgb_width; x++) 
            rows[y][x] = receive_buffer[y][x];
    }
    create_png(path_to_write, info_ptr, rows);
    delete [] row_pointers;
    delete [] conv_matrix;
    delete [] receive_buffer;

    return 0;
    
}
