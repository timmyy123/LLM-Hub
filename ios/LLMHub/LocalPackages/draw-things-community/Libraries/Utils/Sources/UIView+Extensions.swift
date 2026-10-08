#if canImport(UIKit)
  import UIKit

  extension UIView {
    private struct AssociatedKeys {
      static var gradientLayerKey: Void = ()
    }

    private var gradientLayer: CAGradientLayer? {
      get {
        return objc_getAssociatedObject(self, &AssociatedKeys.gradientLayerKey) as? CAGradientLayer
      }
      set {
        objc_setAssociatedObject(
          self, &AssociatedKeys.gradientLayerKey, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
      }
    }

    public func addGradientColors(_ colors: [UIColor], axis: NSLayoutConstraint.Axis)
      -> CAGradientLayer
    {
      // Create a gradient layer
      let gradientLayer = CAGradientLayer()
      gradientLayer.frame = bounds

      // Convert UIColor to CGColor
      gradientLayer.colors = colors.map { $0.cgColor }

      // Set the gradient direction
      switch axis {
      case .vertical:
        gradientLayer.startPoint = CGPoint(x: 0.5, y: 0.0)
        gradientLayer.endPoint = CGPoint(x: 0.5, y: 1.0)
      case .horizontal:
        gradientLayer.startPoint = CGPoint(x: 0.0, y: 0.5)
        gradientLayer.endPoint = CGPoint(x: 1.0, y: 0.5)
      @unknown default:
        break
      }

      // Add the gradient layer to the button
      layer.addSublayer(gradientLayer)
      return gradientLayer
    }

    private func shiftArrayByOne<T>(_ array: [T]) -> [T] {
      guard let lastElement = array.last else { return array }
      var shiftedArray = Array(array.dropLast())
      shiftedArray.insert(lastElement, at: 0)
      return shiftedArray
    }

    public func startGradientAnimation(colors: [UIColor], duration: CFTimeInterval) {
      // Remove any existing gradient layer
      gradientLayer?.removeFromSuperlayer()

      // Create a gradient layer
      let gradientLayer = CAGradientLayer()
      gradientLayer.frame = bounds

      // Convert UIColor to CGColor
      gradientLayer.colors = colors.map { $0.cgColor }

      // Set the gradient direction
      gradientLayer.startPoint = CGPoint(x: 0.0, y: 0.5)
      gradientLayer.endPoint = CGPoint(x: 1.0, y: 0.5)

      // Add the gradient layer to the button
      layer.insertSublayer(gradientLayer, at: 0)

      // Store the gradient layer
      self.gradientLayer = gradientLayer

      var result: [[CGColor]] = []
      var currentColors = colors

      for _ in 0...colors.count {
        currentColors = shiftArrayByOne(currentColors)
        result.append(currentColors.map { $0.cgColor })
      }
      // Create a gradient color animation
      let colorAnimation = CAKeyframeAnimation(keyPath: "colors")
      colorAnimation.duration = duration
      colorAnimation.values = result
      colorAnimation.autoreverses = false
      colorAnimation.repeatCount = .infinity

      // Add the animation to the gradient layer
      gradientLayer.add(colorAnimation, forKey: "colorChangeAnimation")
    }

    public func stopGradientAnimation() {
      gradientLayer?.removeFromSuperlayer()
      gradientLayer?.removeAnimation(forKey: "colorChangeAnimation")
    }

    public func maskWithLabel(_ label: UILabel) {
      // Create a text path from the UILabel
      guard let text = label.text else { return }
      let textPath = CGMutablePath()
      textPath.addRect(bounds)
      let attrString = NSAttributedString(
        string: text, attributes: [NSAttributedString.Key.font: label.font as CTFont])
      let line = CTLineCreateWithAttributedString(attrString)
      let runArray = CTLineGetGlyphRuns(line)

      for runIndex in 0..<CFArrayGetCount(runArray) {
        let run = unsafeBitCast(CFArrayGetValueAtIndex(runArray, runIndex), to: CTRun.self)
        let runFont = CFDictionaryGetValue(
          CTRunGetAttributes(run), unsafeBitCast(kCTFontAttributeName, to: UnsafeRawPointer.self))
        let runFontRef = unsafeBitCast(runFont, to: CTFont.self)

        for glyphIndex in 0..<CTRunGetGlyphCount(run) {
          var glyph = CGGlyph()
          var position = CGPoint()
          CTRunGetGlyphs(run, CFRangeMake(glyphIndex, 1), &glyph)
          CTRunGetPositions(run, CFRangeMake(glyphIndex, 1), &position)

          if let letter = CTFontCreatePathForGlyph(runFontRef, glyph, nil) {
            let labelOrigin = label.frame.origin
            let t = CGAffineTransform(scaleX: 1, y: -1).translatedBy(
              x: position.x + labelOrigin.x,
              y: -(position.y + labelOrigin.y / 2 + label.bounds.height))
            textPath.addPath(letter, transform: t)
          }
        }
      }

      // Create a shape layer using the text path
      let maskLayer = CAShapeLayer()
      maskLayer.path = textPath
      maskLayer.fillRule = .evenOdd
      maskLayer.bounds = bounds
      maskLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)

      // Apply the mask to the view
      layer.mask = maskLayer
    }
  }
#endif
