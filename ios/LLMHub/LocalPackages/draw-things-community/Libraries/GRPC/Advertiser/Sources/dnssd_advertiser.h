#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

void advertise(const char *service_name, const char *service_domain, const char *host_name, uint16_t host_port, uint32_t ttl);
void send_mdns_packet(const char* hostname, const char* ip_address);
char* get_current_ip_address(void);
