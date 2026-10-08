//
//  LocalizedString.swift
//  DrawThings
//
//  Created by wlin1 on 5/1/23.
//

import Foundation

public struct LocalizedString {
  static var localizedStringsDebug = false
  public static var bundle = Bundle.main

  private static var debugLocalizedStringBundle: Bundle? {
    if let path = Bundle.main.path(forResource: "en", ofType: "lproj") {
      return Bundle(path: path)
    }
    return nil
  }

  public static func forKey(_ key: String) -> String {
    var localizedString = NSLocalizedString(key, bundle: bundle, comment: "")
    if localizedStringsDebug {
      let enString = debugLocalizedStringBundle?.localizedString(
        forKey: key, value: nil, table: nil)
      let languageCode = Locale.current.languageCode ?? ""
      if languageCode != "en" && enString == localizedString {

        // example: welcome en-US
        let message = "\(key) - \(Locale.current.identifier)"
        return message
      }
    }
    if let path = Bundle.main.path(forResource: "en", ofType: "lproj"),
      let englishBundle = Bundle(path: path), localizedString == key
    {
      localizedString = englishBundle.localizedString(forKey: key, value: nil, table: nil)
    }
    return localizedString
  }

  public static func format(key: String, _ arguments: CVarArg...) -> String {
    return String(format: LocalizedString.forKey(key), arguments: arguments)
  }
}
