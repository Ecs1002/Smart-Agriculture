#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <string>

#define tcp_in_buff_size        13
#define tcp_out_buff_size       34

#define uart_in_buff_size       34
#define uart_out_buff_size      13

#define cc_str_len 100


int main(){

system("sudo fuser -k /dev/ttyS0");


    //---------- UART_setup ---------//
    int serial_port = open("/dev/ttyS0", O_RDWR);
    struct termios tty;
    if(tcgetattr(serial_port, &tty) != 0) { return 1; }
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;
    tty.c_cc[VTIME] = 10;
    tty.c_cc[VMIN] = 0;
    cfsetispeed(&tty, B921600);
    cfsetospeed(&tty, B921600);
    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) { return 1; }

    uint8_t uart_out_buff[uart_out_buff_size];
    uint8_t uart_in_buff[uart_in_buff_size];
    //-------------------------------//


    usleep(2000);
    std::string cc_str;
    uint8_t cc_chr[cc_str_len];
    uint8_t check_chr[cc_str_len];
    std::ifstream cc_file("/home/pi/compass_cal.txt");
    getline(cc_file, cc_str);
    for(int i=0; i<cc_str_len; i++){ cc_chr[i] = cc_str[i]; }
    cc_file.close();

    write(serial_port, cc_chr, cc_str_len);
    int reasult;
    while(reasult < cc_str_len){ ioctl(serial_port, FIONREAD, &reasult); }
    read(serial_port, &check_chr, cc_str_len);

    for(int i=0; i<cc_str_len; i++){
        if( check_chr[i] != cc_chr[i] ){ std::cout << "Not Matched at index: " << i << "\n"; }
    }

    usleep(1000);


    //---------- TCP_setup ---------//
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    uint8_t tcp_in_buff[tcp_in_buff_size] = { 0 };
    uint8_t  tcp_out_buff[tcp_out_buff_size] = { 0 };

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(server_fd, SOL_SOCKET,SO_REUSEADDR | SO_REUSEPORT, &opt,sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(9191);

    bind(server_fd, (struct sockaddr*)&address,sizeof(address));
    listen(server_fd, 3);
    new_socket = accept(server_fd, (struct sockaddr*)&address,(socklen_t*)&addrlen);
    //------------------------------//


    while(true){

        valread = read(new_socket, tcp_in_buff, tcp_in_buff_size);

        for(int i = 0; i < uart_out_buff_size; i++){ uart_out_buff[i] = tcp_in_buff[i]; }
        write(serial_port, uart_out_buff, uart_out_buff_size);

        int reasult;
        while(reasult < uart_in_buff_size){ ioctl(serial_port, FIONREAD, &reasult); }
        read(serial_port, &uart_in_buff, uart_in_buff_size);

        for(int i = 0; i < uart_in_buff_size; i++){ tcp_out_buff[i] = uart_in_buff[i]; }
        write(new_socket, tcp_out_buff, tcp_out_buff_size);



        if(tcp_in_buff[5]=='!' && tcp_in_buff[6]=='1'){ printf("Shutting Down...\n"); break; }

    }

        system("sudo killall yct");
        close(new_socket);
        close(serial_port);
        system("sudo shutdown now");

    return 0;
}
