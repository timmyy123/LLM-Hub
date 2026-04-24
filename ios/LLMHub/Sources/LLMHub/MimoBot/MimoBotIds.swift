import Foundation
import CoreBluetooth

/// Canonical identifiers shared across the Mimo bot stack.
/// Must stay in sync with docs/mimobot-protocol.md and the Android MimoBotIds.kt.
enum MimoBotIds {
    static let service    = CBUUID(string: "6d696d6f-b07d-4e13-9e88-000000000001")
    static let audioUp    = CBUUID(string: "6d696d6f-b07d-4e13-9e88-000000000010")
    static let audioDown  = CBUUID(string: "6d696d6f-b07d-4e13-9e88-000000000011")
    static let control    = CBUUID(string: "6d696d6f-b07d-4e13-9e88-000000000012")

    static let advNamePrefix = "mimobot-"

    // Audio format (see docs/mimobot-protocol.md)
    static let sampleRateHz  = 16_000
    static let frameMs       = 20
    static let frameSamples  = 320            // sampleRateHz * frameMs / 1000
    static let channels      = 1
    static let opusBitrate   = 24_000
}
