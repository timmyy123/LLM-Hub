import DataModels
import Diffusion
import HTTPServer
import Invocation
import ModelZoo
import NNC
import Utils

public protocol HTTPAPIServerDelegate: AnyObject {
  func generateImages(forInvocation invocation: Invocation) throws -> ([Data], Bool)
  func HTTPRequestParametersFromExistingConfiguration() -> Parameters
}

public final class HTTPAPIServer {
  private let httpServer: HTTPServer
  private let callbackQueue: DispatchQueue
  public weak var delegate: HTTPAPIServerDelegate?

  public init(callbackQueue: DispatchQueue) {
    self.callbackQueue = callbackQueue
    httpServer = HTTPServer()
    httpServer.addPOSTRoute(withPath: "/sdapi/v1/txt2img") { [weak self] data in
      guard let self = self else { return Self.createErrorResponse(error: "") }
      return callbackQueue.sync {
        return self.handleRequest(body: data, imageToImage: false)
      }
    }
    httpServer.addPOSTRoute(withPath: "/sdapi/v1/img2img") { [weak self] data in
      guard let self = self else { return Self.createErrorResponse(error: "") }
      return callbackQueue.sync {
        return self.handleRequest(body: data, imageToImage: true)
      }
    }
    httpServer.addGETRoute(withPath: "/") { [weak self] data in
      guard let self = self else { return Self.createErrorResponse(error: "") }
      return self.welcome(body: data)
    }
    httpServer.addGETRoute(withPath: "/sdapi/v1/options") { [weak self] data in
      guard let self = self else { return Self.createErrorResponse(error: "") }
      return self.welcome(body: data)
    }
  }

  private static func createErrorResponse(error: Error) -> HTTPServerResponse {
    let response = ErrorResponse(error: "HTTPException", detail: "\(error)", body: "", errors: "")
    let data = (try? JSONEncoder().encode(response)) ?? Data()
    print("Returning error response:\n\(String(data: data, encoding: .utf8) ?? "nil")")
    return HTTPServerResponse(body: data, statusCode: 422, mimeType: "application/json")
  }

  private func handleRequest(body: Data, imageToImage: Bool) -> HTTPServerResponse {
    do {
      guard let parameters = delegate?.HTTPRequestParametersFromExistingConfiguration() else {
        return Self.createErrorResponse(error: "")
      }
      let request = try RequestBody.createRequestBody(data: body, parameters: parameters)
      let image: Invocation.PrefersDefaultOptional<Tensor<FloatType>>
      let mask: Invocation.PrefersDefaultOptional<Tensor<UInt8>>
      if imageToImage {
        if let images = request.images,
          let firstImage = images.first,
          images.count == 1
        {
          try Validation.validate(
            firstImage.shape[1] == parameters.heightParameter.value,
            errorMessage:
              "init_images height doesn't match \(firstImage.shape[1]) from the parameter \(parameters.heightParameter.value)"
          )
          try Validation.validate(
            firstImage.shape[2] == parameters.widthParameter.value,
            errorMessage:
              "init_images width doesn't match \(firstImage.shape[2]) from the parameter \(parameters.widthParameter.value)"
          )
          image = .some(firstImage)
          mask = .none
        } else {
          image = .prefersDefault
          mask = .prefersDefault
        }
      } else {
        try Validation.validate(
          request.images == nil, errorMessage: "init_images is not supported for text-to-image")
        image = .none
        mask = .none
      }
      let faceRestorationModel =
        request.restoreFaces ? EverythingZoo.availableSpecifications[0].file : nil
      var resizingOccurred: Bool = false
      if !imageToImage {
        parameters.strengthParameter.value = 1
      }
      let invocation = try Invocation(
        faceRestorationModel: faceRestorationModel,
        image: image, mask: mask, parameters: parameters, resizingOccurred: &resizingOccurred,
        fromHTTPServer: true)
      let (images, _) = try delegate?.generateImages(forInvocation: invocation) ?? ([], false)
      let response = SuccessResponse(images: images)
      let body = try JSONEncoder().encode(response)
      print("Returning success response")  // Ready to send this to Console on editWorkflow.
      return HTTPServerResponse(body: body, statusCode: 200, mimeType: "application/json")
    } catch let error {
      return Self.createErrorResponse(error: error)
    }
  }

  public func bind(ip: String, port: Int) -> Bool {
    return httpServer.bind(toIP: ip, port: Int32(port))
  }

  public func listenAfterBinding() {
    httpServer.listenAfterBinding()
  }

  public func stop() {
    httpServer.stop()
  }

  deinit {
    httpServer.stop()  // HTTPServer.stop() is idempotent, so it's ok if it's already been stopped here
  }
}

extension HTTPAPIServer {
  final class WelcomeResponse: Encodable {
    let parameters: Parameters
    init(parameters: Parameters) {
      self.parameters = parameters
    }
    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: JSONKey.self)
      for parameter in parameters.allParameters() {
        let jsonKey = JSONKey(parameter.commandLineFlag.replacingOccurrences(of: "-", with: "_"))
        try parameter.encode(to: &container, forKey: jsonKey)
      }
    }
  }

  private func welcome(body: Data) -> HTTPServerResponse {
    guard let parameters = delegate?.HTTPRequestParametersFromExistingConfiguration() else {
      return Self.createErrorResponse(error: "")
    }
    let data = (try? JSONEncoder().encode(WelcomeResponse(parameters: parameters))) ?? Data()
    return HTTPServerResponse(body: data, statusCode: 200, mimeType: "application/json")
  }
}
