import Foundation
import MapKit
import EventKit
import UIKit
import AVFoundation
import Contacts
import CoreLocation
import UserNotifications
import CryptoKit
#if canImport(AlarmKit)
import AlarmKit
import SwiftUI
#endif

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

#if canImport(AlarmKit)
@available(iOS 26.0, *)
private struct AgentAlarmMetadata: AlarmMetadata, Codable {}
#endif

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

    private func cleanHTML(_ raw: String) -> String {
        var s = raw
        if let re = try? NSRegularExpression(pattern: "<[^>]*>") {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: " ")
        }
        return s
            .replacingOccurrences(of: "&amp;",  with: "&")
            .replacingOccurrences(of: "&lt;",   with: "<")
            .replacingOccurrences(of: "&gt;",   with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;",  with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func webSearch(query: String) async -> String {
        let cleanQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let encoded = cleanQuery.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) else {
            return "Invalid query."
        }

        // 1. Try DuckDuckGo Instant Answer API
        if let jsonUrl = URL(string: "https://api.duckduckgo.com/?q=\(encoded)&format=json&no_html=1&skip_disambig=1") {
            var req = URLRequest(url: jsonUrl)
            req.setValue("LLM Hub iOS", forHTTPHeaderField: "User-Agent")
            if let (data, _) = try? await URLSession.shared.data(for: req),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                var results: [String] = []
                if let abstract = json["Abstract"] as? String, !abstract.isEmpty {
                    results.append(abstract)
                }
                if let def = json["Definition"] as? String, !def.isEmpty {
                    results.append(def)
                }
                if let topics = json["RelatedTopics"] as? [[String: Any]] {
                    for topic in topics.prefix(3) {
                        if let text = topic["Text"] as? String, !text.isEmpty {
                            results.append(text)
                        }
                    }
                }
                if !results.isEmpty {
                    return results.joined(separator: "\n---\n")
                }
            }
        }

        // 2. DuckDuckGo HTML fallback with cleanHTML
        if let htmlUrl = URL(string: "https://duckduckgo.com/html/?q=\(encoded)") {
            var req = URLRequest(url: htmlUrl)
            req.setValue("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0", forHTTPHeaderField: "User-Agent")
            req.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
            req.setValue("en-US,en;q=0.5", forHTTPHeaderField: "Accept-Language")

            if let (data, resp) = try? await URLSession.shared.data(for: req),
               (resp as? HTTPURLResponse)?.statusCode == 200,
               let html = String(data: data, encoding: .utf8) {
                
                let nsHTML = html as NSString
                let range = NSRange(location: 0, length: nsHTML.length)
                if let sRe = try? NSRegularExpression(pattern: #"<a class="result__snippet"[^>]*>(.*?)</a>"#, options: .dotMatchesLineSeparators) {
                    let matches = sRe.matches(in: html, range: range)
                    var snippets: [String] = []
                    for m in matches.prefix(5) {
                        let rawSnippet = nsHTML.substring(with: m.range(at: 1))
                        let cleaned = cleanHTML(rawSnippet)
                        if !cleaned.isEmpty {
                            snippets.append(cleaned)
                        }
                    }
                    if !snippets.isEmpty {
                        return snippets.joined(separator: "\n---\n")
                    }
                }
            }
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

    // MARK: - Calculate Hash
    
    public func calculateHash(text: String, algorithm: String = "SHA-256") -> String {
        let data = Data(text.utf8)
        let algo = algorithm.uppercased()
        if algo.contains("MD5") {
            let digest = Insecure.MD5.hash(data: data)
            return digest.map { String(format: "%02hhx", $0) }.joined()
        } else if algo.contains("SHA1") || algo.contains("SHA-1") {
            let digest = Insecure.SHA1.hash(data: data)
            return digest.map { String(format: "%02hhx", $0) }.joined()
        } else if algo.contains("512") {
            let digest = SHA512.hash(data: data)
            return digest.map { String(format: "%02hhx", $0) }.joined()
        } else {
            let digest = SHA256.hash(data: data)
            return digest.map { String(format: "%02hhx", $0) }.joined()
        }
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

    // MARK: - Open Map

    @MainActor
    public func openMap(location: String) -> String {
        let loc = location.trimmingCharacters(in: .whitespacesAndNewlines)
        let query = loc.isEmpty ? "current location" : loc
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "maps://?q=\(encoded)") ?? URL(string: "http://maps.apple.com/?q=\(encoded)") else {
            return "Failed to encode location '\(query)'."
        }
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
            return "Opening Apple Maps for '\(query)'."
        }
        return "Apple Maps unavailable on this device."
    }

    // MARK: - Send SMS / Message

    public func resolvePhoneNumber(for recipient: String) async -> String {
        let clean = recipient.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.allSatisfy({ $0.isNumber || $0 == "+" || $0 == "-" || $0 == " " || $0 == "(" || $0 == ")" }) {
            return clean
        }

        return await Task.detached(priority: .userInitiated) {
            let store = CNContactStore()
            let keysToFetch: [CNKeyDescriptor] = [
                CNContactPhoneNumbersKey as CNKeyDescriptor,
                CNContactGivenNameKey as CNKeyDescriptor,
                CNContactFamilyNameKey as CNKeyDescriptor
            ]
            let request = CNContactFetchRequest(keysToFetch: keysToFetch)
            var foundNumber: String? = nil

            do {
                try store.enumerateContacts(with: request) { contact, stop in
                    let fullName = "\(contact.givenName) \(contact.familyName)".trimmingCharacters(in: .whitespacesAndNewlines)
                    if fullName.localizedCaseInsensitiveContains(clean) || contact.givenName.localizedCaseInsensitiveContains(clean) {
                        if let phone = contact.phoneNumbers.first?.value.stringValue {
                            foundNumber = phone
                            stop.pointee = true
                        }
                    }
                }
            } catch {
                print("Error enumerating contacts: \(error)")
            }
            return foundNumber ?? clean
        }.value
    }

    @MainActor
    public func sendSms(recipient: String, body: String) async -> String {
        let targetNumber = await resolvePhoneNumber(for: recipient)
        let cleanBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let encodedBody = cleanBody.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "sms:\(targetNumber)?body=\(encodedBody)") ?? URL(string: "sms:\(targetNumber)") else {
            return "Failed to prepare message for '\(recipient)' (\(targetNumber))."
        }
        if UIApplication.shared.canOpenURL(url) {
            await UIApplication.shared.open(url)
            return "Opening Messages to send '\(cleanBody)' to \(recipient) (\(targetNumber))..."
        }
        return "SMS app unavailable on this device."
    }

// MARK: - Location Helper (GPS Coordinates)

@MainActor
public class AgentLocationHelper: NSObject, @preconcurrency CLLocationManagerDelegate {
    public static let shared = AgentLocationHelper()
    private let manager = CLLocationManager()
    public var lastLocation: CLLocationCoordinate2D?
    private var locationContinuations: [CheckedContinuation<CLLocationCoordinate2D?, Never>] = []

    public var currentCoordinate: CLLocationCoordinate2D? {
        if let loc = manager.location?.coordinate, loc.latitude != 0 && loc.longitude != 0 {
            return loc
        }
        return lastLocation
    }

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = kCLDistanceFilterNone
        if manager.authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()
        }
        manager.startUpdatingLocation()
    }

    public func requestGPSCoordinate() async -> CLLocationCoordinate2D? {
        if let coord = currentCoordinate {
            return coord
        }

        if manager.authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()
        }

        manager.startUpdatingLocation()
        manager.requestLocation()

        return await withCheckedContinuation { continuation in
            if let coord = self.currentCoordinate {
                continuation.resume(returning: coord)
            } else {
                self.locationContinuations.append(continuation)
                Task { @MainActor [weak self] in
                    try? await Task.sleep(nanoseconds: 3_500_000_000)
                    guard let self = self else { return }
                    if let idx = self.locationContinuations.firstIndex(where: { $0 as AnyObject === continuation as AnyObject }) {
                        let c = self.locationContinuations.remove(at: idx)
                        c.resume(returning: self.lastLocation)
                    }
                }
            }
        }
    }

    public func getBestCoordinate() async -> CLLocationCoordinate2D? {
        // 1. Actively request & wait for real iOS CoreLocation hardware GPS fix
        if let gpsCoord = await requestGPSCoordinate() {
            return gpsCoord
        }

        // 2. IP Geolocation Fallback 1: ip-api.com
        if let url = URL(string: "http://ip-api.com/json"),
           let (data, _) = try? await URLSession.shared.data(from: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let status = json["status"] as? String, status == "success",
           let lat = json["lat"] as? Double, let lon = json["lon"] as? Double {
            let coord = CLLocationCoordinate2D(latitude: lat, longitude: lon)
            lastLocation = coord
            return coord
        }

        // 3. IP Geolocation Fallback 2: ipapi.co
        if let url = URL(string: "https://ipapi.co/json/"),
           let (data, _) = try? await URLSession.shared.data(from: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let lat = json["latitude"] as? Double, let lon = json["longitude"] as? Double {
            let coord = CLLocationCoordinate2D(latitude: lat, longitude: lon)
            lastLocation = coord
            return coord
        }

        // 4. IP Geolocation Fallback 3: ipinfo.io
        if let url = URL(string: "https://ipinfo.io/json"),
           let (data, _) = try? await URLSession.shared.data(from: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let locStr = json["loc"] as? String {
            let parts = locStr.components(separatedBy: ",")
            if parts.count == 2, let lat = Double(parts[0]), let lon = Double(parts[1]) {
                let coord = CLLocationCoordinate2D(latitude: lat, longitude: lon)
                lastLocation = coord
                return coord
            }
        }

        return nil
    }

    public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let loc = locations.last?.coordinate, loc.latitude != 0 && loc.longitude != 0 {
            lastLocation = loc
            let continuations = locationContinuations
            locationContinuations.removeAll()
            for c in continuations {
                c.resume(returning: loc)
            }
        }
    }

    public func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("LocationManager error: \(error)")
        let continuations = locationContinuations
        locationContinuations.removeAll()
        for c in continuations {
            c.resume(returning: lastLocation)
        }
    }

    public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        if manager.authorizationStatus == .authorizedWhenInUse || manager.authorizationStatus == .authorizedAlways {
            manager.startUpdatingLocation()
            manager.requestLocation()
        }
    }
}

    // MARK: - Geocode / Local POI Search

    @MainActor
    public func geocodeLocation(_ location: String) async -> (Double, Double, String)? {
        let query = location.trimmingCharacters(in: CharacterSet(charactersIn: "\"'\t\n\r "))
        guard !query.isEmpty else { return nil }

        // Try MKLocalSearch first (Apple Native POI Search with GPS region bias)
        let searchRequest = MKLocalSearch.Request()
        searchRequest.naturalLanguageQuery = query
        
        let userCoord = AgentLocationHelper.shared.currentCoordinate
        if let coord = userCoord {
            searchRequest.region = MKCoordinateRegion(center: coord, latitudinalMeters: 50000, longitudinalMeters: 50000)
        }

        let localSearch = MKLocalSearch(request: searchRequest)
        if let response = try? await localSearch.start(), let firstItem = response.mapItems.first {
            let coord = firstItem.placemark.coordinate
            let name = firstItem.name ?? firstItem.placemark.title ?? query
            return (coord.latitude, coord.longitude, name)
        }

        // Fallback to Nominatim OpenStreetMap API with GPS coordinates if available
        var urlStr = "https://nominatim.openstreetmap.org/search?q=\(query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")&format=json&limit=1"
        if let coord = userCoord {
            urlStr += "&lat=\(coord.latitude)&lon=\(coord.longitude)"
        }
        guard let url = URL(string: urlStr) else { return nil }

        var request = URLRequest(url: url)
        request.setValue("LLMHub-App", forHTTPHeaderField: "User-Agent")
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            if let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]],
               let first = arr.first,
               let latStr = first["lat"] as? String, let lat = Double(latStr),
               let lonStr = first["lon"] as? String, let lon = Double(lonStr) {
                let name = (first["display_name"] as? String) ?? query
                return (lat, lon, name)
            }
        } catch {
            print("Geocoding error: \(error)")
        }
        return nil
    }

    /// Search for a category (e.g. "bar", "cafe") near the user's current GPS location.
    /// Uses MKLocalSearch with a 15km radius, falling back to Nominatim viewbox.
    @MainActor
    public func geocodeNearby(category: String) async -> (Double, Double, String)? {
        guard let coord = await AgentLocationHelper.shared.getBestCoordinate() else { return nil }
        let query = category.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return nil }

        // MKLocalSearch with tight 15 km region centred on user
        let searchRequest = MKLocalSearch.Request()
        searchRequest.naturalLanguageQuery = query
        searchRequest.region = MKCoordinateRegion(center: coord, latitudinalMeters: 15000, longitudinalMeters: 15000)
        let localSearch = MKLocalSearch(request: searchRequest)
        if let response = try? await localSearch.start(), let firstItem = response.mapItems.first {
            let c = firstItem.placemark.coordinate
            let name = firstItem.name ?? firstItem.placemark.title ?? query
            return (c.latitude, c.longitude, name)
        }

        // Nominatim viewbox ~15 km around user
        let delta = 0.15
        let minLon = coord.longitude - delta
        let minLat = coord.latitude - delta
        let maxLon = coord.longitude + delta
        let maxLat = coord.latitude + delta
        let viewbox = "\(minLon),\(minLat),\(maxLon),\(maxLat)"
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let vbEncoded = viewbox.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? viewbox
        let urlStr = "https://nominatim.openstreetmap.org/search?q=\(encoded)&format=json&limit=1&bounded=1&viewbox=\(vbEncoded)"
        guard let url = URL(string: urlStr) else { return nil }
        var request = URLRequest(url: url)
        request.setValue("LLMHub-App", forHTTPHeaderField: "User-Agent")
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            if let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]],
               let first = arr.first,
               let latStr = first["lat"] as? String, let lat = Double(latStr),
               let lonStr = first["lon"] as? String, let lon = Double(lonStr) {
                let name = (first["display_name"] as? String) ?? query
                return (lat, lon, name)
            }
        } catch {
            print("geocodeNearby error: \(error)")
        }

        // Nothing found nearby — show user's location with the category as label
        return (coord.latitude, coord.longitude, "\(query) near you")
    }

    // MARK: - Calendar & Weather & Alarm Tools

    @MainActor
    public func addCalendarEvent(title: String, dateStr: String) async -> String {
        let eventStore = EKEventStore()
        let granted: Bool
        if #available(iOS 17.0, *) {
            granted = (try? await eventStore.requestWriteOnlyAccessToEvents()) ?? false
        } else {
            granted = await withCheckedContinuation { continuation in
                eventStore.requestAccess(to: .event) { ok, _ in continuation.resume(returning: ok) }
            }
        }
        guard granted else { return "Calendar permission denied." }

        let event = EKEvent(eventStore: eventStore)
        event.title = title.isEmpty ? "New Event" : title
        let targetDate = Date().addingTimeInterval(86400)
        event.startDate = targetDate
        event.endDate = targetDate.addingTimeInterval(3600)
        event.calendar = eventStore.defaultCalendarForNewEvents

        do {
            try eventStore.save(event, span: .thisEvent)
            return "Successfully added '\(event.title!)' to calendar for tomorrow."
        } catch {
            return "Failed to save calendar event: \(error.localizedDescription)"
        }
    }

    @MainActor
    public func checkWeather(location: String) async -> String {
        var clean = location.trimmingCharacters(in: CharacterSet(charactersIn: "()\"' \t\n\r"))
        if let firstQuote = clean.firstIndex(of: "\""), let lastQuote = clean.lastIndex(of: "\""), firstQuote < lastQuote {
            clean = String(clean[clean.index(after: firstQuote)..<lastQuote])
        }

        let lower = clean.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let isCurrentLocationQuery = lower.isEmpty || lower == "query" || lower == "weather" || lower == "my location" ||
            lower == "current location" || lower == "here" || lower == "me" || lower == "nearby" || lower == "local" ||
            lower.contains("weather") || lower.contains("current location") || lower.contains("my location") || lower.contains("here")

        var targetLat: Double?
        var targetLon: Double?
        var targetName: String?

        if isCurrentLocationQuery {
            if let coord = await AgentLocationHelper.shared.getBestCoordinate() {
                targetLat = coord.latitude
                targetLon = coord.longitude
            }
        }

        if targetLat == nil || targetLon == nil {
            if !isCurrentLocationQuery, let (lat, lon, name) = await geocodeLocation(clean) {
                targetLat = lat
                targetLon = lon
                targetName = name
            } else if let coord = await AgentLocationHelper.shared.getBestCoordinate() {
                targetLat = coord.latitude
                targetLon = coord.longitude
            }
        }

        guard let lat = targetLat, let lon = targetLon else {
            return "Could not determine location for weather check."
        }

        let name: String
        if let targetName = targetName {
            name = targetName
        } else if let rev = await geocodeReverse(lat: lat, lon: lon) {
            name = rev
        } else {
            name = "your location"
        }
        let urlStr = "https://api.open-meteo.com/v1/forecast?latitude=\(lat)&longitude=\(lon)&current_weather=true"
        guard let url = URL(string: urlStr),
              let (data, _) = try? await URLSession.shared.data(from: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let current = json["current_weather"] as? [String: Any],
              let temp = current["temperature"] as? Double,
              let wind = current["windspeed"] as? Double else {
            return "Weather info for '\(name)' is currently unavailable."
        }
        return "Weather in \(name): \(temp)°C, Wind: \(wind) km/h."
    }

    @MainActor
    private func geocodeReverse(lat: Double, lon: Double) async -> String? {
        let loc = CLLocation(latitude: lat, longitude: lon)
        let geocoder = CLGeocoder()
        if let placemarks = try? await geocoder.reverseGeocodeLocation(loc), let pm = placemarks.first {
            return pm.locality ?? pm.subAdministrativeArea ?? pm.name
        }
        return nil
    }

    private func parseTargetDate(from text: String) -> (date: Date, timeStr: String, desc: String) {
        let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let now = Date()
        let calendar = Calendar.current

        // 1. Check for relative minutes (e.g. "1 minute", "in 5 minutes", "10 mins", "1 minute from now")
        if let minRegex = try? NSRegularExpression(pattern: "(\\d+)\\s*(?:min|minute)s?", options: .caseInsensitive),
           let match = minRegex.firstMatch(in: lower, options: [], range: NSRange(location: 0, length: lower.utf16.count)),
           let r = Range(match.range(at: 1), in: lower),
           let mins = Double(lower[r]) {
            let seconds = mins * 60
            let targetDate = now.addingTimeInterval(seconds)
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            let timeStr = formatter.string(from: targetDate)
            let desc = mins == 1 ? "1 minute from now" : "\(Int(mins)) minutes from now"
            return (targetDate, timeStr, desc)
        }

        // 2. Check for relative seconds (e.g. "30 seconds", "in 45 secs")
        if let secRegex = try? NSRegularExpression(pattern: "(\\d+)\\s*(?:sec|second)s?", options: .caseInsensitive),
           let match = secRegex.firstMatch(in: lower, options: [], range: NSRange(location: 0, length: lower.utf16.count)),
           let r = Range(match.range(at: 1), in: lower),
           let secs = Double(lower[r]) {
            let targetDate = now.addingTimeInterval(secs)
            let formatter = DateFormatter()
            formatter.timeStyle = .medium
            let timeStr = formatter.string(from: targetDate)
            return (targetDate, timeStr, "\(Int(secs)) seconds from now")
        }

        // 3. Check for relative hours (e.g. "1 hour", "in 2 hours")
        if let hrRegex = try? NSRegularExpression(pattern: "(\\d+)\\s*(?:hr|hour)s?", options: .caseInsensitive),
           let match = hrRegex.firstMatch(in: lower, options: [], range: NSRange(location: 0, length: lower.utf16.count)),
           let r = Range(match.range(at: 1), in: lower),
           let hrs = Double(lower[r]) {
            let targetDate = now.addingTimeInterval(hrs * 3600)
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            let timeStr = formatter.string(from: targetDate)
            return (targetDate, timeStr, "\(Int(hrs)) hours from now")
        }

        // 4. Absolute time (e.g. "7:00 AM", "19:30", "7:30")
        let timeComponents = parseTimeComponents(from: lower)
        let targetHour = timeComponents.hour ?? 21
        let targetMinute = timeComponents.minute ?? 0
        
        var fullComponents = calendar.dateComponents([.year, .month, .day], from: now)
        fullComponents.hour = targetHour
        fullComponents.minute = targetMinute
        fullComponents.second = 0
        
        var targetDate = calendar.date(from: fullComponents) ?? now.addingTimeInterval(60)
        if targetDate <= now {
            targetDate = calendar.date(byAdding: .day, value: 1, to: targetDate) ?? targetDate
        }
        
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        let timeStr = formatter.string(from: targetDate)
        return (targetDate, timeStr, timeStr)
    }

    private func parseTimeComponents(from text: String) -> DateComponents {
        let lower = text.lowercased()
        var dateComponents = DateComponents()
        
        let pattern = "(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?"
        if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
           let match = regex.firstMatch(in: lower, options: [], range: NSRange(location: 0, length: lower.utf16.count)) {
            
            if let hRange = Range(match.range(at: 1), in: lower), let rawH = Int(lower[hRange]) {
                var hour = rawH
                var minuteVal = 0
                if match.range(at: 2).location != NSNotFound,
                   let mRange = Range(match.range(at: 2), in: lower),
                   let parsedM = Int(lower[mRange]) {
                    minuteVal = parsedM
                }
                
                var isPM = false
                var isAM = false
                if match.range(at: 3).location != NSNotFound,
                   let ampmRange = Range(match.range(at: 3), in: lower) {
                    let str = lower[ampmRange]
                    isPM = str == "pm"
                    isAM = str == "am"
                }
                
                if isPM && hour < 12 { hour += 12 }
                if isAM && hour == 12 { hour = 0 }
                
                dateComponents.hour = Swift.min(Swift.max(hour, 0), 23)
                dateComponents.minute = Swift.min(Swift.max(minuteVal, 0), 59)
            }
        }
        
        if dateComponents.hour == nil {
            dateComponents.hour = 21
            dateComponents.minute = 0
        }
        
        return dateComponents
    }

    @MainActor
    public func setAlarm(time: String, label: String) async -> String {
        let cleanTime = time.isEmpty ? "1 minute" : time
        let cleanLabel = label.isEmpty ? "Alarm" : label

        let (targetDate, timeStr, desc) = parseTargetDate(from: cleanTime)
        let interval = max(1.0, targetDate.timeIntervalSinceNow)

        let center = UNUserNotificationCenter.current()
        let options: UNAuthorizationOptions = [.alert, .sound, .badge]
        let granted = (try? await center.requestAuthorization(options: options)) ?? false

        if granted {
            let content = UNMutableNotificationContent()
            content.title = "⏰ \(cleanLabel)"
            content.body = "Alarm set for \(timeStr) (\(desc))"
            content.sound = UNNotificationSound.defaultCriticalSound(withAudioVolume: 1.0)
            content.interruptionLevel = .timeSensitive

            let trigger = UNTimeIntervalNotificationTrigger(timeInterval: interval, repeats: false)
            let request = UNNotificationRequest(identifier: "alarm-\(UUID().uuidString)", content: content, trigger: trigger)
            try? await center.add(request)
            return "Alarm '\(cleanLabel)' scheduled for \(timeStr) (\(desc)). Notification sound will alert you."
        }

        if let clockUrl = URL(string: "clock-alarm:"), UIApplication.shared.canOpenURL(clockUrl) {
            await UIApplication.shared.open(clockUrl)
            return "Opened Clock app. Please set alarm for \(timeStr)."
        }

        return "Alarm notification permission denied. Enable notifications in iOS Settings."
    }
}
