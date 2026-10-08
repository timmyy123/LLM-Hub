import Foundation
import Logging

public protocol GRPCServiceBrowserDelegate: AnyObject {
  func didFindService(_: GRPCServiceBrowser.ServiceDescriptor)
  func didRemoveService(_: GRPCServiceBrowser.ServiceDescriptor)
}

public class GRPCServiceBrowser {
  public struct ServiceDescriptor: Equatable & Hashable {
    public var name: String
    public var host: String
    public var port: Int
    public var TLS: Bool?
    public init(name: String, host: String, port: Int, TLS: Bool?) {
      self.name = name
      self.host = host
      self.port = port
      self.TLS = TLS
    }
  }

  private var serviceBrowser: NetServiceBrowser
  private var discoveredServices = [NetService]()
  private var discoveredDescriptors = [ServiceDescriptor]()
  public weak var delegate: GRPCServiceBrowserDelegate?
  let logger = Logger(label: "com.draw-things.grpc-service")

  private let objCResponder: ObjCResponder

  final class ObjCResponder: NSObject, NetServiceBrowserDelegate, NetServiceDelegate {
    weak var serviceBrowser: GRPCServiceBrowser? = nil
    public func netServiceBrowser(
      _ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool
    ) {
      serviceBrowser?.netServiceBrowser(browser, didFind: service, moreComing: moreComing)
    }
    public func netServiceBrowser(
      _ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool
    ) {
      serviceBrowser?.netServiceBrowser(browser, didRemove: service, moreComing: moreComing)
    }
    public func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) {
      serviceBrowser?.netService(sender, didNotResolve: errorDict)
    }
    public func netServiceDidResolveAddress(_ sender: NetService) {
      serviceBrowser?.netServiceDidResolveAddress(sender)
    }
  }
  public init() {
    serviceBrowser = NetServiceBrowser()
    objCResponder = ObjCResponder()
    objCResponder.serviceBrowser = self
    serviceBrowser.delegate = objCResponder
    serviceBrowser.searchForServices(ofType: "_dt-grpc._tcp.", inDomain: "local.")
  }

  private static func nameAndTLS(name serviceName: String) -> (name: String, TLS: Bool?) {
    let TLS: Bool?
    let name: String
    if serviceName.hasSuffix("_notls") {
      TLS = false
      name = String(serviceName.prefix(upTo: serviceName.index(serviceName.endIndex, offsetBy: -6)))
    } else if serviceName.hasSuffix("_tls") {
      TLS = true
      name = String(serviceName.prefix(upTo: serviceName.index(serviceName.endIndex, offsetBy: -4)))
    } else {
      TLS = nil
      name = serviceName
    }
    return (name: name, TLS: TLS)
  }

  // NetServiceBrowserDelegate method
  public func netServiceBrowser(
    _ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool
  ) {
    dispatchPrecondition(condition: .onQueue(.main))
    discoveredServices.append(service)
    let (name, TLS) = Self.nameAndTLS(name: service.name)
    discoveredDescriptors.append(
      ServiceDescriptor(name: name, host: service.hostName ?? "", port: service.port, TLS: TLS))
    service.delegate = objCResponder
    service.resolve(withTimeout: 5.0)
  }

  public func netServiceBrowser(
    _ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool
  ) {
    dispatchPrecondition(condition: .onQueue(.main))
    guard let firstIndex = discoveredServices.firstIndex(where: { $0 == service }) else {
      return
    }
    let descriptor = discoveredDescriptors[firstIndex]
    discoveredServices.remove(at: firstIndex)
    discoveredDescriptors.remove(at: firstIndex)
    delegate?.didRemoveService(descriptor)
  }

  // NetServiceDelegate method
  public func netServiceDidResolveAddress(_ sender: NetService) {
    dispatchPrecondition(condition: .onQueue(.main))
    if let addresses = sender.addresses, !addresses.isEmpty {
      if let hostname = sender.hostName, sender.port != 0 {
        let (name, TLS) = Self.nameAndTLS(name: sender.name)
        let descriptor = ServiceDescriptor(name: name, host: hostname, port: sender.port, TLS: TLS)
        if let firstIndex = discoveredServices.firstIndex(where: { $0 == sender }) {
          discoveredDescriptors[firstIndex] = descriptor  // Update the descriptor.
        }
        delegate?.didFindService(descriptor)
      }
    }
  }

  public func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) {
    dispatchPrecondition(condition: .onQueue(.main))
    logger.error("Failed to resolve service: \(errorDict)")
    if let hostName = sender.hostName, sender.port != 0 {
      let (_, TLS) = Self.nameAndTLS(name: sender.name)
      if let (decodedHostname, ipAddress) = GRPCHostnameUtils.decodeGRPCHostname(
        customizeHostname: hostName)
      {
        let descriptor = ServiceDescriptor(
          name: decodedHostname, host: ipAddress, port: sender.port, TLS: TLS)
        if let firstIndex = discoveredServices.firstIndex(where: { $0 == sender }) {
          discoveredDescriptors[firstIndex] = descriptor
          logger.info("fallback hostname: \(hostName) ipaddress:\(ipAddress)")
        }
      }

    }
  }

}
