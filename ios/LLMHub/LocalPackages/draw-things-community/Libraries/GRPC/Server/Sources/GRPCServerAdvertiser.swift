import Foundation

#if canImport(Network)
  import Network
#endif
#if canImport(C_DNSAdvertiser)
  import C_DNSAdvertiser

  private func localhost() -> String? {
    let maxLength = Int(HOST_NAME_MAX) + 1
    var buffer = [CChar](repeating: 0, count: maxLength)

    if gethostname(&buffer, maxLength) == 0 {
      return String(cString: buffer)
    } else {
      return nil
    }
  }

  private func IP() -> String? {
    guard let string = get_current_ip_address() else {
      return nil
    }
    let result = String(cString: string)
    string.deallocate()
    return result
  }
#endif

public class GRPCServerAdvertiser {
  private let name: String

  #if canImport(C_DNSAdvertiser)
    enum AdvertisingPhase {
      case none
      case probe(Int)
      case announcement(Int)
      case goodbye
    }
    private let queue: DispatchQueue
    private var phase: AdvertisingPhase = .none
  #else
    private var netService: NetService?
    private let objCResponder: ObjCResponder

    final class ObjCResponder: NSObject, NetServiceDelegate {
      weak var advertiser: GRPCServerAdvertiser? = nil
      public func netServiceDidPublish(_ sender: NetService) {
        advertiser?.netServiceDidPublish(sender)
      }
      public func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) {
        advertiser?.netService(sender, didNotPublish: errorDict)
      }
    }
  #endif

  public init(name: String) {
    self.name = name
    #if canImport(C_DNSAdvertiser)
      queue = DispatchQueue(label: "com.draw-things.advertise", qos: .default)
    #else
      objCResponder = ObjCResponder()
      objCResponder.advertiser = self
    #endif
  }

  private var nameHostnameIPAndPort: (String, String, String, UInt16)?

  public func startAdvertising(port: Int32, TLS: Bool) {
    let name = TLS ? "\(name)_tls" : "\(name)_notls"
    #if canImport(C_DNSAdvertiser)
      if let IP = IP(), let localhost = localhost() {
        let hostname = localhost + ".local"
        nameHostnameIPAndPort = (name, hostname, IP, UInt16(port))
        queue.async { [weak self] in
          guard let self = self else { return }
          self.phase = .probe(0)
          self.advertise(name: name, hostname: hostname, IP: IP, port: UInt16(port))
        }
      }
    #else
      let serviceType = "_dt-grpc._tcp."
      let serviceDomain = "local."
      let netService = NetService(domain: serviceDomain, type: serviceType, name: name, port: port)
      self.netService = netService
      netService.delegate = objCResponder
      netService.publish()  // Removed the options parameter
    #endif
  }

  public func stopAdvertising(completionHandler: (() -> Void)? = nil) {
    #if canImport(C_DNSAdvertiser)
      guard let (name, hostname, IP, port) = nameHostnameIPAndPort else { return }
      if let completionHandler = completionHandler {
        queue.async {
          self.phase = .goodbye
          self.advertise(name: name, hostname: hostname, IP: IP, port: port)
          completionHandler()
        }
      } else {
        queue.sync {
          phase = .goodbye
          advertise(name: name, hostname: hostname, IP: IP, port: port)
        }
      }
    #else
      netService?.stop()
      netService = nil
    #endif
  }

}

extension GRPCServerAdvertiser {
  #if canImport(C_DNSAdvertiser)
    private static let probeCount = 3
    private static let announcementIntervals = [1, 2, 4, 8, 16]
    func advertise(name: String, hostname: String, IP: String, port: UInt16) {
      dispatchPrecondition(condition: .onQueue(queue))
      switch phase {
      case .none:
        return  // Do nothing.
      case .probe(let count):
        send_mdns_packet(hostname, IP)
        if count + 1 < Self.probeCount {
          phase = .probe(count + 1)
        } else {
          phase = .announcement(0)
        }
        queue.asyncAfter(deadline: .now() + .milliseconds(250)) { [weak self] in
          guard let self = self else { return }
          self.advertise(name: name, hostname: hostname, IP: IP, port: port)
        }
      case .announcement(let count):
        C_DNSAdvertiser.advertise(name, "_dt-grpc._tcp.local", hostname, port, 120)
        phase = .announcement(count + 1)
        if count < Self.announcementIntervals.count {
          queue.asyncAfter(deadline: .now() + .seconds(Self.announcementIntervals[count])) {
            [weak self] in
            guard let self = self else { return }
            self.advertise(name: name, hostname: hostname, IP: IP, port: port)
          }
        } else {
          queue.asyncAfter(deadline: .now() + .seconds(60)) { [weak self] in
            guard let self = self else { return }
            self.advertise(name: name, hostname: hostname, IP: IP, port: port)
          }
        }
      case .goodbye:
        C_DNSAdvertiser.advertise(name, "_dt-grpc._tcp.local", hostname, port, 0)
        phase = .none
      }
    }
  #else
    // NetServiceDelegate methods
    func netServiceDidPublish(_ sender: NetService) {
      print(
        "Service published: domain=\(sender.domain) type=\(sender.type) name=\(sender.name) port=\(sender.port)"
      )
    }

    func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) {
      print("Failed to publish service: \(errorDict)")
    }
  #endif
}
