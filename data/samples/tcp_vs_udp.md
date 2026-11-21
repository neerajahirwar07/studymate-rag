# TCP vs UDP

TCP is a connection-oriented transport protocol. It sets up a connection with a three-way handshake (SYN, SYN-ACK, ACK), delivers bytes in order, retransmits lost packets, and uses flow control and congestion control. These guarantees make TCP reliable but slower.

UDP is connectionless. It sends independent datagrams without a handshake, without ordering and without retransmission. Because it has very little overhead, UDP is used for DNS lookups, video streaming, online games and voice calls, where speed matters more than perfect delivery.

HTTP and HTTPS normally run on top of TCP. HTTP/3 runs over QUIC, which is built on UDP and adds its own reliability.
