import Foundation
import MapKit
import EventKit
import UIKit
import AVFoundation

public struct MapResult {
    public let label: String
    public let latitude: Double
    public let longitude: Double

    public init(label: String, latitude: Double, longitude: Double) {
        self.label = label
        self.latitude = latitude
        self.longitude = longitude
    }
}

@MainActor
public class AgentTools {

    public static let shared = AgentTools()

    private init() {}

    // MARK: - Show Map (MapKit Geocoding)

    public func showMap(location: String) async -> MapResult? {
        let geocoder = CLGeocoder()
        do {
            let placemarks = try await geocoder.geocodeAddressString(location)
            if let mark = placemarks.first, let loc = mark.location {
                let name = mark.name ?? mark.locality ?? location
                return MapResult(label: name, latitude: loc.coordinate.latitude, longitude: loc.coordinate.longitude)
            }
        } catch {
            print("Map geocoding failed: \(error.localizedDescription)")
        }
        return nil
    }

    // MARK: - Query Wikipedia

    public func queryWikipedia(topic: String, lang: String = "en") async -> String {
        guard let encoded = topic.trimmingCharacters(in: .whitespacesAndNewlines).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              let url = URL(string: "https://\(lang).wikipedia.org/api/rest_v1/page/summary/\(encoded)") else {
            return "Invalid topic query."
        }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let extract = json["extract"] as? String {
                return extract
            }
        } catch {
            return "Wikipedia query failed: \(error.localizedDescription)"
        }
        return "No Wikipedia article found for '\(topic)'."
    }

    // MARK: - Web Search (DuckDuckGo)

    public func webSearch(query: String) async -> String {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://html.duckduckgo.com/html/?q=\(encoded)") else {
            return "Invalid query."
        }

        var request = URLRequest(url: url)
        request.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", forHTTPHeaderField: "User-Agent")

        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            if let html = String(data: data, encoding: .utf8) {
                let matches = html.components(separatedBy: "<a class=\"result__snippet")
                var snippets: [String] = []
                for match in matches.dropFirst().prefix(3) {
                    if let textEnd = match.range(of: "</a>") {
                        let rawSnippet = String(match[..<textEnd.lowerBound])
                        let clean = rawSnippet.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        if !clean.isEmpty { snippets.append(clean) }
                    }
                }
                return snippets.joined(separator: "\n---\n")
            }
        } catch {
            return "Web search failed: \(error.localizedDescription)"
        }
        return "No search results found."
    }

    // MARK: - Weather (wttr.in)

    public func getCurrentWeather(location: String) async -> String {
        guard let encoded = location.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://wttr.in/\(encoded)?format=3") else {
            return "Invalid location."
        }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
                return text
            }
        } catch {
            return "Weather query failed: \(error.localizedDescription)"
        }
        return "Weather unavailable for '\(location)'."
    }

    // MARK: - Create Calendar Event (EventKit)

    public func createCalendarEvent(title: String, location: String = "", notes: String = "") async -> String {
        let store = EKEventStore()
        do {
            let granted = try await store.requestWriteOnlyAccessToEvents()
            if granted {
                let event = EKEvent(eventStore: store)
                event.title = title
                event.location = location
                event.notes = notes
                event.startDate = Date().addingTimeInterval(3600)
                event.endDate = Date().addingTimeInterval(7200)
                event.calendar = store.defaultCalendarForNewEvents
                try store.save(event, span: .thisEvent)
                return "Event '\(title)' created in Calendar."
            }
        } catch {
            return "Calendar permission or creation error: \(error.localizedDescription)"
        }
        return "Calendar access denied."
    }

    // MARK: - Send Email

    @MainActor
    public func sendEmail(email: String, subject: String, body: String) -> String {
        guard let encodedSub = subject.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let encodedBody = body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "mailto:\(email)?subject=\(encodedSub)&body=\(encodedBody)") else {
            return "Invalid email URL."
        }
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
            return "Opened Mail app for '\(email)'."
        }
        return "Mail app not available."
    }

    // MARK: - Send SMS

    @MainActor
    public func sendSms(phone: String, body: String) -> String {
        guard let encodedBody = body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "sms:\(phone)&body=\(encodedBody)") else {
            return "Invalid SMS URL."
        }
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
            return "Opened Messages app for '\(phone)'."
        }
        return "Messages app not available."
    }

    // MARK: - Run Apple Shortcut

    @MainActor
    public func runAppleShortcut(name: String) -> String {
        guard let encoded = name.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "shortcuts://run-shortcut?name=\(encoded)") else {
            return "Invalid shortcut name."
        }
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
            return "Triggered Shortcut '\(name)'."
        }
        return "Shortcuts app not available."
    }

    // MARK: - Toggle Flashlight

    @MainActor
    public func toggleFlashlight(enabled: Bool) -> String {
        guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else {
            return "Flashlight unavailable on this device."
        }
        do {
            try device.lockForConfiguration()
            device.torchMode = enabled ? .on : .off
            device.unlockForConfiguration()
            return "Flashlight turned \(enabled ? "on" : "off")."
        } catch {
            return "Failed to toggle flashlight: \(error.localizedDescription)"
        }
    }
}
