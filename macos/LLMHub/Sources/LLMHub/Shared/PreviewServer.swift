import Foundation
import Darwin
import os.log

private let log = Logger(subsystem: "com.llmhub.llmhub", category: "PreviewServer")

// Loopback-only HTTP server using BSD POSIX sockets.
// Does NOT use the Network framework, so it does NOT trigger the
// iOS local network permission dialog.

public actor LocalHTMLPreviewServer {
    public static let shared = LocalHTMLPreviewServer()

    private var serverFD: Int32 = -1

    public func stop() {
        if serverFD >= 0 {
            log.debug("[PreviewServer] stop() closing fd=\(self.serverFD)")
            Darwin.close(serverFD)
            serverFD = -1
        } else {
            log.debug("[PreviewServer] stop() called but no server running")
        }
    }

    public func start(html: String) throws -> URL {
        log.debug("[PreviewServer] start() called, html size=\(html.count) chars")
        stop()

        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            log.error("[PreviewServer] socket() failed errno=\(errno)")
            throw POSIXError(.ENOTSOCK)
        }
        log.debug("[PreviewServer] socket() fd=\(fd)")

        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        // Bind to 127.0.0.1 on an OS-chosen port
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = CFSwapInt32HostToBig(UInt32(INADDR_LOOPBACK))
        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            let e = errno
            log.error("[PreviewServer] bind() failed errno=\(e) (\(String(cString: strerror(e))))")
            Darwin.close(fd)
            throw POSIXError(POSIXErrorCode(rawValue: e) ?? .EADDRINUSE)
        }

        guard Darwin.listen(fd, 5) == 0 else {
            let e = errno
            log.error("[PreviewServer] listen() failed errno=\(e)")
            Darwin.close(fd)
            throw POSIXError(POSIXErrorCode(rawValue: e) ?? .ECONNREFUSED)
        }

        // Read back the assigned port
        var boundAddr = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &boundAddr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &len)
            }
        }
        let port = CFSwapInt16BigToHost(boundAddr.sin_port)
        log.debug("[PreviewServer] listening on 127.0.0.1:\(port) fd=\(fd)")

        serverFD = fd

        // Accept loop on background thread
        let htmlCopy = html
        Thread.detachNewThread {
            Self.acceptLoop(fd: fd, html: htmlCopy)
        }

        guard let url = URL(string: "http://127.0.0.1:\(port)/") else {
            throw POSIXError(.EINVAL)
        }
        log.debug("[PreviewServer] returning URL \(url.absoluteString)")
        return url
    }

    private static func acceptLoop(fd: Int32, html: String) {
        log.debug("[PreviewServer] acceptLoop started fd=\(fd)")
        while true {
            let clientFD = Darwin.accept(fd, nil, nil)
            if clientFD < 0 {
                log.debug("[PreviewServer] acceptLoop exiting (accept returned \(clientFD), errno=\(errno))")
                break
            }
            log.debug("[PreviewServer] accepted clientFD=\(clientFD)")
            let htmlCopy = html
            Thread.detachNewThread {
                Self.handle(clientFD: clientFD, html: htmlCopy)
            }
        }
    }

    private static func handle(clientFD: Int32, html: String) {
        defer {
            log.debug("[PreviewServer] closing clientFD=\(clientFD)")
            Darwin.close(clientFD)
        }

        // Read request (we don't need to parse it)
        var buf = [UInt8](repeating: 0, count: 4096)
        let recvd = Darwin.recv(clientFD, &buf, buf.count, 0)
        log.debug("[PreviewServer] recv \(recvd) bytes on clientFD=\(clientFD)")

        let bodyData = html.data(using: .utf8) ?? Data()
        let header = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: \(bodyData.count)\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
        var response = Data()
        response.append(header.data(using: .utf8)!)
        response.append(bodyData)
        response.withUnsafeBytes { ptr in
            let sent = Darwin.send(clientFD, ptr.baseAddress!, ptr.count, 0)
            log.debug("[PreviewServer] sent \(sent)/\(ptr.count) bytes to clientFD=\(clientFD)")
        }
    }
}
