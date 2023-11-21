#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <raspicam/raspicam.h>

#define PORT 6666
int width = 640;
int height = 480;

void RGB2Yuv420p(uint8_t *destination, uint8_t *rgb, size_t width, size_t height){
    size_t image_size = width * height;
    size_t upos = image_size;
    size_t vpos = upos + upos / 4;
    size_t i = 0;

    for( size_t line = 0; line < height; ++line )
    {
        if( !(line % 2) )
        {
            for( size_t x = 0; x < width; x += 2 )
            {
                uint8_t r = rgb[3 * i];
                uint8_t g = rgb[3 * i + 1];
                uint8_t b = rgb[3 * i + 2];

                destination[i++] = (uint8_t)(0.299*r + 0.587*g + 0.114*b);

                destination[upos++] = (uint8_t)( (-0.169*r - 0.331*g + 0.499*b) + 128 );
                destination[vpos++] = (uint8_t)( (0.499*r - 0.418*g - 0.0813*b) + 128 );




                r = rgb[3 * i];
                g = rgb[3 * i + 1];
                b = rgb[3 * i + 2];

                destination[i++] = (uint8_t)(0.299*r + 0.587*g + 0.114*b);
            }
        }
        else
        {
            for( size_t x = 0; x < width; x += 1 )
            {
                uint8_t r = rgb[3 * i];
                uint8_t g = rgb[3 * i + 1];
                uint8_t b = rgb[3 * i + 2];

                destination[i++] = (uint8_t)(0.299*r + 0.587*g + 0.114*b);
            }
        }
    }
}

int main(){
        //============ Initialize_Camera ===========//
        raspicam::RaspiCam Camera;
        Camera.setFormat(raspicam::RASPICAM_FORMAT_RGB);
        Camera.open();
        Camera.setHeight(height);
        Camera.setWidth(width);
        //==========================================//

        //======== Initialize_Compression ==========//;
        uint32_t raw_res = width*height;
        uint32_t rgb_img_buf_len = raw_res*3;
        uint8_t rgb_raw_buff[rgb_img_buf_len];
        uint32_t yuv_len = int(raw_res*1.5);
        uint8_t yuv_buff[yuv_len];
        //==========================================//

        //============== Initialize_TCP ============//
        int server_fd, new_socket, valread;
        struct sockaddr_in address;
        int opt = 1;
        int addrlen = sizeof(address);
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(server_fd, SOL_SOCKET,SO_REUSEADDR | SO_REUSEPORT, &opt,sizeof(opt));
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(PORT);
        bind(server_fd, (struct sockaddr*)&address,sizeof(address));
                listen(server_fd, 3);
                new_socket = accept(server_fd, (struct sockaddr*)&address,(socklen_t*)&addrlen);
        //==========================================//
        while(true){
                //=================== Capture_Image =================//
                Camera.grab();
                Camera.retrieve(rgb_raw_buff);
                //===================================================//


                //============== Compress_Image_Buffer ==============//
                RGB2Yuv420p(yuv_buff, rgb_raw_buff, width, height);
                //===================================================//


                //================ TCP_Communication ================//
                write(new_socket, yuv_buff, yuv_len);
                //===================================================//
        }
        return 0;
}



