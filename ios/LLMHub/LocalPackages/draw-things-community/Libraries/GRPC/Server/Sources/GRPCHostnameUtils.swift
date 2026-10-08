import Foundation

public struct GRPCHostnameUtils {

  // create customize grpc host name will encode grpc ip address
  // in order to fallback if user only receive the hostname without ip
  // 192.168 -> a, 10.0 -> b, others x.x -> x.x.c
  // bob, 192.168.0.1 -> bob.a.0.1.local, bob, 10.0.0.1 -> bob.b.0.1.local, fallback alice, 100.200.1.1 -> alice.local
  public static func customizeGRPCHostname(hostname: String, ip: String) -> String {
    // length limit from DNS RFC 1035
    let maxHostnameLength = Int(NI_MAXHOST)

    let ipComponents = ip.split(separator: ".")
    guard ipComponents.count == 4 else {
      return hostname
    }
    var hostname = hostname
    let nameSuffix: String
    if ipComponents[0] == "192", ipComponents[1] == "168" {
      nameSuffix = ".a.\(ipComponents[2]).\(ipComponents[3]).local"
    } else if ipComponents[0] == "10", ipComponents[1] == "0" {
      nameSuffix = ".b.\(ipComponents[2]).\(ipComponents[3]).local"
    } else {
      nameSuffix = ".local"  // do nothing but attach .local for local dns broadcast
    }

    let maxAllowedHostnameLength = maxHostnameLength - nameSuffix.count
    if hostname.count > maxAllowedHostnameLength {
      let endIndex = hostname.index(hostname.startIndex, offsetBy: maxAllowedHostnameLength)
      hostname = String(hostname[..<endIndex])
    }
    return "\(hostname)\(nameSuffix)"
  }

  // decode customize grpc host name will back to original host name and ip address
  public static func decodeGRPCHostname(customizeHostname: String) -> (
    hostname: String, ip: String
  )? {
    let components = customizeHostname.split(separator: ".")
    guard components.count >= 6 else {
      return nil
    }

    // Extract hostname components (all parts before the letter)
    let letterIndex = components.count - 4
    let letter = components[letterIndex]

    let ipFirstOctet: String
    let hostnameComponents: ArraySlice<Substring>
    if letter == "b" {
      ipFirstOctet = "10.0"
      hostnameComponents = components[0..<letterIndex]
    } else if letter == "a" {
      ipFirstOctet = "192.168"
      hostnameComponents = components[0..<letterIndex]
    } else {
      return nil
    }

    let ipAddress = "\(ipFirstOctet).\(components[letterIndex + 1]).\(components[letterIndex + 2])"
    let hostname = hostnameComponents.joined(separator: ".")

    return (hostname: hostname, ip: ipAddress)
  }

}
