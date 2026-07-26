import Foundation
import MapKit
import EventKit
import UIKit
import AVFoundation
import Contacts
import CoreLocation
import UserNotifications
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

    public var currentCoordinate: CLLocationCoordinate2D? {
        if let loc = manager.location?.coordinate {
            return loc
        }
        return lastLocation
    }

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
    }

    public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let loc = locations.last {
            lastLocation = loc.coordinate
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
        let queryLoc: String
        if lower.isEmpty || lower == "weather" || lower.contains("weather") || lower.contains("current location") || lower.contains("my location") || lower.contains("here") || lower.hasPrefix("location") {
            queryLoc = "Melbourne"
        } else {
            queryLoc = clean
        }

        guard let (lat, lon, name) = await geocodeLocation(queryLoc) else {
            return "Could not determine location for weather check."
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
        let cleanTime = time.isEmpty ? "9:00 PM" : time
        let cleanLabel = label.isEmpty ? "Alarm" : label
        
        let timeComponents = parseTimeComponents(from: cleanTime)
        let targetHour = timeComponents.hour ?? 21
        let targetMinute = timeComponents.minute ?? 0
        let formattedTime = String(format: "%02d:%02d", targetHour, targetMinute)

        #if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let alarmManager = AlarmManager.shared
                let authState = try await alarmManager.requestAuthorization()
                if authState == .authorized {
                    let alarmTime = Alarm.Schedule.Relative.Time(hour: targetHour, minute: targetMinute)
                    let schedule: Alarm.Schedule = .relative(.init(time: alarmTime, repeats: .never))

                    let titleResource = LocalizedStringResource(stringLiteral: cleanLabel)
                    let stopBtnResource = LocalizedStringResource(stringLiteral: "Stop")
                    let stopButton = AlarmButton(text: stopBtnResource, textColor: .red, systemImageName: "stop.circle")
                    let alertContent = AlarmPresentation.Alert(
                        title: titleResource,
                        stopButton: stopButton
                    )
                    let presentation = AlarmPresentation(alert: alertContent)
                    let attributes = AlarmAttributes<AgentAlarmMetadata>(presentation: presentation, metadata: AgentAlarmMetadata(), tintColor: .blue)

                    let alarmID = UUID()
                    let configuration = AlarmManager.AlarmConfiguration<AgentAlarmMetadata>(
                        schedule: schedule,
                        attributes: attributes
                    )

                    _ = try await alarmManager.schedule(id: alarmID, configuration: configuration)
                    return "Alarm '\(cleanLabel)' set for \(formattedTime)."
                }
            } catch {
                print("AlarmKit request failed (\(error.localizedDescription)), using notification fallback.")
            }
        }
        #endif

        let calendar = Calendar.current
        let now = Date()
        var fullComponents = calendar.dateComponents([.year, .month, .day], from: now)
        fullComponents.hour = targetHour
        fullComponents.minute = targetMinute
        fullComponents.second = 0
        
        if let candidate = calendar.date(from: fullComponents), candidate <= now {
            if let tomorrow = calendar.date(byAdding: .day, value: 1, to: now) {
                let tmrw = calendar.dateComponents([.year, .month, .day], from: tomorrow)
                fullComponents.year = tmrw.year
                fullComponents.month = tmrw.month
                fullComponents.day = tmrw.day
            }
        }

        let center = UNUserNotificationCenter.current()
        let options: UNAuthorizationOptions = [.alert, .sound, .badge]
        let granted = (try? await center.requestAuthorization(options: options)) ?? false

        if granted {
            let content = UNMutableNotificationContent()
            content.title = "⏰ \(cleanLabel)"
            content.body = "Alarm: \(formattedTime)"
            content.sound = UNNotificationSound.defaultCriticalSound(withAudioVolume: 1.0)
            content.interruptionLevel = .timeSensitive
            
            let trigger = UNCalendarNotificationTrigger(dateMatching: fullComponents, repeats: false)
            let request = UNNotificationRequest(identifier: "alarm-\(UUID().uuidString)", content: content, trigger: trigger)
            try? await center.add(request)
        }

        if let clockUrl = URL(string: "clock-alarm:"), UIApplication.shared.canOpenURL(clockUrl) {
            await UIApplication.shared.open(clockUrl)
        }
        
        return "Alarm '\(cleanLabel)' set for \(formattedTime)."
    }
}
