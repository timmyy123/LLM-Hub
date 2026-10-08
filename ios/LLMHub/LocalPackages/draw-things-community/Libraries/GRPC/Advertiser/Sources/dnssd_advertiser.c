#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/in.h>
#include <net/if.h>
#include <netdb.h>
#include <ifaddrs.h>

#define MDNS_MULTICAST_ADDR "224.0.0.251"
#define MDNS_PORT 5353

// Encode DNS name for mDNS
void encode_dns_name(char *buffer, const char *name) {
    char *pos = buffer;
    const char *start = name;
    const char *dot;

    while ((dot = strchr(start, '.'))) {
        *pos++ = dot - start;
        memcpy(pos, start, dot - start);
        pos += dot - start;
        start = dot + 1;
    }

    *pos++ = strlen(start);  // Last part before null byte
    memcpy(pos, start, strlen(start));
    pos += strlen(start);
    *pos++ = 0;  // End with null byte
}

// Send mDNS packet
void advertise(const char *service_name, const char *service_domain, const char *host_name, uint16_t host_port, uint32_t ttl_param) {
    int sockfd;
    struct sockaddr_in addr;
    char packet[512];
    int pos = 0;

    // Create the UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Multicast address for mDNS
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(MDNS_MULTICAST_ADDR);  // mDNS multicast address
    addr.sin_port = htons(MDNS_PORT);

    // DNS Header
    uint16_t transaction_id = 0;
    uint16_t flags = htons(0x8400);  // Response, authoritative answer
    uint16_t questions = 0;
    uint16_t answers = htons(4);  // 4 Answer Records: PTR, PTR, SRV, TXT
    uint16_t authority = 0;
    uint16_t additional = htons(6);  // 6 additional records (for cache flushes)

    memcpy(packet + pos, &transaction_id, sizeof(transaction_id)); pos += 2;
    memcpy(packet + pos, &flags, sizeof(flags)); pos += 2;
    memcpy(packet + pos, &questions, sizeof(questions)); pos += 2;
    memcpy(packet + pos, &answers, sizeof(answers)); pos += 2;
    memcpy(packet + pos, &authority, sizeof(authority)); pos += 2;
    memcpy(packet + pos, &additional, sizeof(additional)); pos += 2;

    // PTR Record (_grpc._tcp.local -> DrawThings._grpc._tcp.local)
    encode_dns_name(packet + pos, service_domain); pos += strlen(packet + pos) + 1;
    uint16_t type_ptr = htons(12);  // PTR record
    uint16_t class_in = htons(0x8001);  // IN class + Cache flush
    uint32_t ttl = htonl(ttl_param);  // Time to live (2 minutes)
    size_t service_name_len = strlen(service_name);
    size_t service_domain_len = strlen(service_domain);
    uint16_t rdlength = htons(service_name_len + 1 + service_domain_len + 2);
    memcpy(packet + pos, &type_ptr, sizeof(type_ptr)); pos += 2;
    memcpy(packet + pos, &class_in, sizeof(class_in)); pos += 2;
    memcpy(packet + pos, &ttl, sizeof(ttl)); pos += 4;
    memcpy(packet + pos, &rdlength, sizeof(rdlength)); pos += 2;
	char dns_name[service_name_len + 1 + service_domain_len + 1];
	memcpy(dns_name, service_name, service_name_len);
	dns_name[service_name_len] = '.';
	memcpy(dns_name + service_name_len + 1, service_domain, service_domain_len);
	dns_name[service_name_len + 1 +  service_domain_len] = 0;
    encode_dns_name(packet + pos, dns_name); pos += strlen(packet + pos) + 1;

    // SRV Record (DrawThings._dt-grpc._tcp.local -> Port 3819)
    encode_dns_name(packet + pos, dns_name); pos += strlen(packet + pos) + 1;
    uint16_t type_srv = htons(33);  // SRV record
    uint16_t srv_class_in = htons(0x8001);  // IN class + Cache flush
    uint16_t priority = htons(0);
    uint16_t weight = htons(0);
    uint16_t port = htons(host_port);
    uint16_t srv_rdlength = htons(6 + strlen(host_name) + 2);  // Adjusted for the new domain name
    memcpy(packet + pos, &type_srv, sizeof(type_srv)); pos += 2;
    memcpy(packet + pos, &srv_class_in, sizeof(srv_class_in)); pos += 2;
    memcpy(packet + pos, &ttl, sizeof(ttl)); pos += 4;
    memcpy(packet + pos, &srv_rdlength, sizeof(srv_rdlength)); pos += 2;
    memcpy(packet + pos, &priority, sizeof(priority)); pos += 2;
    memcpy(packet + pos, &weight, sizeof(weight)); pos += 2;
    memcpy(packet + pos, &port, sizeof(port)); pos += 2;
    encode_dns_name(packet + pos, host_name); pos += strlen(packet + pos) + 1;


    // TXT Record (for service info)
    encode_dns_name(packet + pos, dns_name); pos += strlen(packet + pos) + 1;
    uint16_t type_txt = htons(16);  // TXT record
    uint16_t txt_class_in = htons(0x8001);  // IN class + Cache flush
    uint16_t txt_rdlength = htons(5);  // Length of the text (e.g., "txtvers=1")
    char txt_data[] = "\x09txtvers=1";  // Simple TXT record
    memcpy(packet + pos, &type_txt, sizeof(type_txt)); pos += 2;
    memcpy(packet + pos, &txt_class_in, sizeof(txt_class_in)); pos += 2;
    memcpy(packet + pos, &ttl, sizeof(ttl)); pos += 4;
    memcpy(packet + pos, &txt_rdlength, sizeof(txt_rdlength)); pos += 2;
    memcpy(packet + pos, txt_data, sizeof(txt_data) - 1); pos += sizeof(txt_data) - 1;

    // Send the packet
    int packet_len = pos;
    if (sendto(sockfd, packet, packet_len, 0, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Sendto failed");
    } else {
        printf("mDNS packet sent successfully\n");
    }

    close(sockfd);
}

// Function to construct and send an mDNS packet to register a hostname
void send_mdns_packet(const char* hostname, const char* ip_address) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Failed to create socket");
        exit(EXIT_FAILURE);
    }

    // Set the socket to allow multicast
    int ttl = 1;
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
        perror("Failed to set multicast TTL");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    struct sockaddr_in mdns_addr;
    memset(&mdns_addr, 0, sizeof(mdns_addr));
    mdns_addr.sin_family = AF_INET;
    mdns_addr.sin_port = htons(MDNS_PORT);  // Use defined MDNS_PORT
    inet_pton(AF_INET, MDNS_MULTICAST_ADDR, &mdns_addr.sin_addr);  // Use defined MDNS_MULTICAST_ADDR

    // Example mDNS response packet (simplified)
    unsigned char packet[512] = {0};
    int packet_len = 0;

    // Set the transaction ID to 0 (mDNS uses 0 as transaction ID)
    packet[packet_len++] = 0;
    packet[packet_len++] = 0;

    // Set flags: response (bit 15), authoritative answer (bit 10)
    packet[packet_len++] = 0x84;  // Flags high byte
    packet[packet_len++] = 0x00;  // Flags low byte

    // Questions count = 0
    packet[packet_len++] = 0;
    packet[packet_len++] = 0;

    // Answer Record Count = 1 (we are sending one answer for the hostname)
    packet[packet_len++] = 0;
    packet[packet_len++] = 1;

    // Authority and Additional Record Count = 0
    packet[packet_len++] = 0;
    packet[packet_len++] = 0;
    packet[packet_len++] = 0;
    packet[packet_len++] = 0;

    // Add the hostname in DNS label format (e.g., "myhost.local")
    char *label = strdup(hostname);
    char *token = strtok(label, ".");
    
    while (token) {
        size_t token_len = strlen(token);
        packet[packet_len++] = (unsigned char) token_len;
        memcpy(&packet[packet_len], token, token_len);
        packet_len += token_len;
        token = strtok(NULL, ".");
    }
    free(label);
    
    packet[packet_len++] = 0; // Null-terminate the DNS name

    // Set the record type to A (IPv4 address)
    packet[packet_len++] = 0x00;
    packet[packet_len++] = 0x01;  // Type A

    // Set the class to IN (Internet)
    packet[packet_len++] = 0x00;
    packet[packet_len++] = 0x01;

    // Set the TTL (Time to Live) to 120 seconds
    packet[packet_len++] = 0x00;
    packet[packet_len++] = 0x00;
    packet[packet_len++] = 0x00;
    packet[packet_len++] = 0x78;

    // Set the data length (IPv4 address size: 4 bytes)
    packet[packet_len++] = 0x00;
    packet[packet_len++] = 0x04;

    // Convert the IP address string (e.g., "192.168.1.100") to binary format and add it to the packet
    struct in_addr ip;
    if (inet_pton(AF_INET, ip_address, &ip) != 1) {
        perror("Invalid IP address format");
        return;
    }

    // Append the 4 bytes of the IPv4 address to the packet
    memcpy(&packet[packet_len], &ip, sizeof(ip));
    packet_len += sizeof(ip);

    // Send the packet to the mDNS multicast address
    if (sendto(sockfd, packet, packet_len, 0, (struct sockaddr*)&mdns_addr, sizeof(mdns_addr)) < 0) {
        perror("Failed to send mDNS packet");
    } else {
        printf("mDNS packet sent to register hostname: %s with IP: %s\n", hostname, ip_address);
    }
    close(sockfd);
}

char* get_current_ip_address() {
    struct ifaddrs *ifaddr, *ifa;
    int family, s;
    char host[NI_MAXHOST];
    char *address = NULL;

    // Get the list of network interfaces
    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return NULL;
    }

    // Iterate through the list of interfaces
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) continue;

        // Check the interface flags to see if it is running and not a loopback interface
        int flags = ifa->ifa_flags;
        int isRunning = (flags & (IFF_UP | IFF_RUNNING)) == (IFF_UP | IFF_RUNNING);
        int isLoopback = (flags & IFF_LOOPBACK) != 0;

        if (isRunning && !isLoopback) {
            family = ifa->ifa_addr->sa_family;

            // Check for IPv4 or IPv6
            if (family == AF_INET || family == AF_INET6) {
                s = getnameinfo(ifa->ifa_addr,
                                (family == AF_INET) ? sizeof(struct sockaddr_in) :
                                                      sizeof(struct sockaddr_in6),
                                host, NI_MAXHOST,
                                NULL, 0, NI_NUMERICHOST);
                if (s == 0) {
                    if (family == AF_INET) {
                        // Prefer IPv4 address over IPv6
                        address = strdup(host);
                        break;
                    } else if (address == NULL) {
                        address = strdup(host);
                    }
                }
            }
        }
    }

    // Free the interface list memory
    freeifaddrs(ifaddr);

    return address;
}

